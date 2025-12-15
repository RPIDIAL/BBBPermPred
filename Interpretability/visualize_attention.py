# predict_attn.py
import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from BBBPermPred.Interpretability.attn_utils import wire_attention_recorder, plot_attention
from BBBPermPred.Models.ChemformerClassifier import ChemformerClassifier  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _build_id2tok_from_chemformer(tokenizer):
    """
    Works with your ChemformerTokenizer:
      - If tokenizer.vocabulary is a LIST (id -> token), use it as a table.
      - If tokenizer.vocabulary is a DICT (token -> id), invert it.
    Uses tokenizer.special_tokens['pad'] to derive pad idx if present.
    Returns: id2tok(ids_list_or_tensor) -> [str], pad_idx (or None)
    """
    vocab = getattr(tokenizer, "vocabulary", None)
    specials = getattr(tokenizer, "special_tokens", {})
    pad_tok = specials.get("pad", None)

    if isinstance(vocab, list):
        table = vocab
        pad_idx = table.index(pad_tok) if (pad_tok in table) else None
        def id2tok(ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            out = []
            for i in ids:
                if pad_idx is not None and i == pad_idx:
                    out.append("<PAD>")
                elif 0 <= int(i) < len(table):
                    out.append(table[int(i)])
                else:
                    out.append(f"<UNK:{int(i)}>") 
            return out
        return id2tok, pad_idx

    if isinstance(vocab, dict):
        id_to_token = {v: k for k, v in vocab.items()}
        pad_idx = vocab.get(pad_tok, None) if pad_tok is not None else None
        def id2tok(ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            out = []
            for i in ids:
                if pad_idx is not None and i == pad_idx:
                    out.append("<PAD>")
                else:
                    out.append(id_to_token.get(int(i), f"<UNK:{int(i)}>"))
            return out
        return id2tok, pad_idx

    raise ValueError("Chemformer tokenizer has unexpected `vocabulary` type.")

@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(args: DictConfig):
    # Init Chemformer (uses your existing config)
    chemformer = ChemformerClassifier(args)
    chemformer.model.eval()
    chemformer.model.to(DEVICE)

    # Build id->token from the live tokenizer
    tokenizer = chemformer.tokenizer  # exposed by your Chemformer
    id2tok, pad_idx = _build_id2tok_from_chemformer(tokenizer)

    # Get dataloader exactly as in your classify()
    dataloader = chemformer.get_dataloader(args.dataset_part)

    # Where to store attention dumps
    save_dir = "Attention_Visualization"
    os.makedirs(save_dir, exist_ok=True)

    max_batches = 1   # keep small for first run
    visualize_first = True

    embeddings_list, labels_list, preds_list, probs_list = [], [], [], []
    batches_done = 0
    first_plot_done = False

    for b_idx, batch in enumerate(dataloader):
        # Move batch to device like your classify()
        batch = chemformer.on_device(batch)

        with torch.no_grad():
            enc_in = batch["encoder_input"]              # (T,B)
            pad_mask_BT = batch["encoder_pad_mask"].transpose(0, 1)  # (B,T)

            # monkey-patch encoder to record attention for THIS forward only
            unwire = wire_attention_recorder(chemformer.model.encoder)

            # forward encoder
            embs = chemformer.model._construct_input(enc_in)  # (T,B,d_model)
            enc_out = chemformer.model.encoder(embs, src_key_padding_mask=pad_mask_BT)  # (T,B,d_model)

            # pooled embeddings (same as your classify)
            pooled = enc_out.mean(dim=0)  # (B,d_model)

            # classifier
            logits = chemformer.model.classifier(pooled)  # (B, 1 or 2)
            if logits.size(-1) == 1:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze(-1)
                probs_np = torch.cat([1 - probs, probs], dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()

            # restore original MHA forwards and fetch attention
            unwire()
            records = getattr(chemformer.model.encoder, "_attn_records", [])

        # collect outputs
        embeddings_list.append(pooled.cpu().numpy())
        labels_list.append(batch["labels"].cpu().numpy())
        preds_list.append(preds.cpu().numpy())
        probs_list.append(probs_np)

        # save attention per layer: (B,H,T,S)
        for li, rec in enumerate(records):
            A_BHTS = rec["weights"].numpy()
            np.save(os.path.join(save_dir, f"batch{b_idx}_layer{li}.npy"), A_BHTS)

        # save token strings (one line per item in batch)
        B = enc_in.shape[1]
        token_lines = []
        for bi in range(B):
            ids = enc_in[:, bi].detach().cpu()
            token_lines.append(" ".join(id2tok(ids)))
        with open(os.path.join(save_dir, f"batch{b_idx}_tokens.txt"), "w") as f:
            for line in token_lines:
                f.write(line + "\n")

        # (optional) quick visualization for first item/layer/head
        if visualize_first and not first_plot_done and len(records) > 0:
            A = records[-1]["weights"][0, 0]  # last layer, head 0, first item -> (T,S)
            key_pad = batch["encoder_pad_mask"][:, 0].detach().cpu().numpy().astype(bool)  # (T,)
            query_pad = key_pad.copy()
            plot_attention(
                A,
                tokens=token_lines[0].split(),
                key_pad_mask=key_pad,
                query_pad_mask=query_pad,
                title=f"batch{b_idx}: layer {len(records)-1}, head 0"
            )
            first_plot_done = True

        batches_done += 1
        if batches_done >= max_batches:
            break

    # concatenate for metrics/saving
    embs = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)

    outdir = getattr(args, "output_dir", None)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "embeddings.npy"), embs)
        np.save(os.path.join(outdir, "labels.npy"), labels)
        np.save(os.path.join(outdir, "preds.npy"), preds)
        np.save(os.path.join(outdir, "probs.npy"), probs)

    print(f"Saved attention to: {save_dir}")

if __name__ == "__main__":
    main()
