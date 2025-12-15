# molbart/utils/attn_recorder.py
import numpy as np
import torch
import matplotlib.pyplot as plt

import inspect
import torch

def wire_attention_recorder(encoder):
    """
    Monkey-patch each layer's self_attn.forward to capture attention weights.
    After a forward pass + calling unwire(), you'll find:
        encoder._attn_records = [{'layer': i, 'weights': Tensor}, ...]
    The recorded 'weights' may be:
      - (B, H, T, S)  if your torch supports per-head capture, or
      - (T, S) / (B, T, S) if your torch only returns averaged weights.
    """
    records, originals = [], []

    for li, layer in enumerate(encoder.layers):
        mha = layer.self_attn
        orig_forward = mha.forward
        originals.append((mha, orig_forward))

        # Detect supported kwargs on this PyTorch
        sig = inspect.signature(orig_forward)
        accepts = set(sig.parameters.keys())

        def wrapped_forward(query, key, value, **kwargs):
            # Always ask for weights if supported
            if 'need_weights' in accepts:
                kwargs['need_weights'] = True

            # Only pass average_attn_weights if this torch supports it
            if 'average_attn_weights' in accepts:
                kwargs['average_attn_weights'] = False  # keep per-head if possible

            # Ensure we pass the correct names for masks
            # (older torch uses 'attn_mask' and 'key_padding_mask')
            # We leave whatever caller provided in **kwargs alone.

            out, attn_w = orig_forward(query, key, value, **kwargs)  # must return tuple

            # Normalize/annotate shapes for downstream use
            # Common cases:
            #  - Per-head: (B, H, T, S)  (newer torch with average_attn_weights=False)
            #  - Averaged per-head: (T, S) or (B, T, S)  (older torch)
            if isinstance(attn_w, torch.Tensor):
                if attn_w.dim() == 3:
                    # Might be (B*H, T, S) on some builds — try to reshape if possible
                    # Only reshape if it cleanly divides by num_heads
                    BH, T, S = attn_w.shape
                    H = getattr(mha, 'num_heads', None)
                    if H and BH % H == 0:
                        B = BH // H
                        try:
                            attn_w = attn_w.view(B, H, T, S)
                        except Exception:
                            pass  # leave as-is if reshape fails
                # If dim()==2 (T,S) or dim()==3 (B,T,S) averaged weights, we just store them as-is.

            records.append({'layer': li, 'weights': attn_w.detach().cpu()})
            return out, attn_w  # IMPORTANT: preserve original return type (tuple)

        mha.forward = wrapped_forward

    def unwire():
        for mha, orig in originals:
            mha.forward = orig
        encoder._attn_records = records

    encoder._attn_records = records
    return unwire

def plot_attention(attn_TS, tokens=None, key_pad_mask=None, query_pad_mask=None,
                   title="Attention", figsize=(7,5), save_path=None):
    """
    attn_TS: (T,S) tensor/ndarray for one head (queries x keys)
    tokens: list[str] for axis labels (optional)
    key_pad_mask/query_pad_mask: bool arrays where True indicates PAD (to blank rows/cols)
    """
    A = attn_TS.detach().cpu().float().numpy() if isinstance(attn_TS, torch.Tensor) else np.asarray(attn_TS, np.float32)
    if key_pad_mask is not None:
        A[:, np.asarray(key_pad_mask, bool)] = np.nan
    if query_pad_mask is not None:
        A[np.asarray(query_pad_mask, bool), :] = np.nan

    plt.figure(figsize=figsize)
    plt.imshow(A, aspect='auto')
    plt.title(title)
    plt.xlabel('Keys'); plt.ylabel('Queries')
    plt.colorbar(label='Weight')
    if tokens is not None:
        S = A.shape[1]
        xt = np.arange(min(S, len(tokens)))
        plt.xticks(xt, tokens[:len(xt)], rotation=90)
        if len(tokens) >= A.shape[0]:
            yt = np.arange(A.shape[0])
            plt.yticks(yt, tokens[:A.shape[0]])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=180)
    plt.show()
