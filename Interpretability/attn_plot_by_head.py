# plot_per_head_masked_allticks.py
# Run:  python plot_per_head_masked_allticks.py
# It masks specials, renormalizes rows, and shows tokens on EVERY subplot.

import os
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# ===================== CONFIG (EDIT) =====================
TOKENS_PATH = "Attention_Visualization/batch0_tokens.txt"   # space-separated tokens, one line per item
ATTN_PATH   = "Attention_Visualization/batch0_layer5.npy"   # .npy attention for the layer you want
BATCH_INDEX = 0                                   # which line in TOKENS_PATH

# Specials to remove before plotting
CLS_TOKEN = "^"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "&"          # set to None if you don't use EOS
TRIM_AFTER_EOS = True    # cut at first EOS (inclusive) before masking
REMOVE_EOS = True        # also remove EOS from plot

# Grid options
HEADS_TO_SHOW = "all"    # "all" or list like [0,3,5]
N_COLS = 4               # columns in grid of heads

# Label density (1 = label every token; 2 = every 2nd token, ...)
LABEL_EVERY_X = 1
LABEL_EVERY_Y = 1
TICK_FONTSIZE = 8

RENORMALIZE_ROWS = True  # re-softmax each row after masking specials
OUT_DIR = "Attention_Visualization/figs"
# =========================================================

def load_tokens(tokens_path, batch_index=0):
    with open(tokens_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if batch_index >= len(lines):
        raise IndexError(f"batch_index {batch_index} out of range ({len(lines)} lines)")
    return lines[batch_index].split()

def load_attention(npy_path):
    return np.load(npy_path)

def extract_heads(A, batch_index=0):
    """
    Returns a list of (T,S) arrays, one per head.
    Accepts (B,H,T,S), (B,T,S), or (T,S).
    """
    if A.ndim == 4:  # (B,H,T,S)
        return [A[batch_index, h].astype(float) for h in range(A.shape[1])]
    if A.ndim == 3:  # (B,T,S)
        return [A[batch_index].astype(float)]
    if A.ndim == 2:  # (T,S)
        return [A.astype(float)]
    raise ValueError(f"Unexpected attention shape {A.shape}")

def trim_to_first_eos(tokens, mats, eos="&", keep_eos=True):
    if eos is None or eos not in tokens:
        return tokens, mats
    idx = tokens.index(eos)
    cutoff = idx + (1 if keep_eos else 0)
    t2 = tokens[:cutoff]
    out = []
    for M in mats:
        M2 = M[:, :cutoff]
        if M2.shape[0] >= len(tokens):   # self-attn typical
            M2 = M2[:cutoff, :]
        out.append(M2)
    return t2, out

def mask_specials(tokens, mats, specials_cols, specials_rows):
    """
    Remove specials from columns (keys) and (if aligned) from rows (queries).
    Returns trimmed tokens and trimmed matrices (same length for both axes).
    """
    tokens = list(tokens)
    S = min(len(tokens), mats[0].shape[1])
    tokens = tokens[:S]
    keep_cols = np.array([t not in specials_cols for t in tokens], bool)
    keep_rows = np.array([t not in specials_rows for t in tokens], bool)

    tokens_new = [t for t, k in zip(tokens, keep_cols) if k]
    trimmed = []
    for M in mats:
        Mc = M[:, :S][:, keep_cols]
        if Mc.shape[0] >= len(tokens):  # self-attn -> drop same rows
            Mr = Mc[keep_rows[:Mc.shape[0]], :]
        else:
            Mr = Mc
        trimmed.append(Mr)
    return tokens_new, trimmed

def renorm_rows(M):
    den = np.nansum(M, axis=1, keepdims=True)
    den[den == 0] = 1.0
    return M / den

def _tick_positions(n, every):
    idx = np.arange(0, n, max(1, int(every)))
    return idx

def plot_heads_grid_all_ticks(head_mats, tokens, title_prefix, out_png, ncols=4,
                              every_x=1, every_y=1, tick_fs=8):
    H = len(head_mats)
    ncols = min(ncols, max(1, H))
    nrows = int(np.ceil(H / ncols))

    # consistent color scale across heads (robust to outliers)
    all_vals = np.concatenate([m.ravel() for m in head_mats if m.size > 0])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols, 3.2*nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for h, M in enumerate(head_mats):
        ax = axes[h]; ax.axis('on')
        im = ax.imshow(M, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"Head {h}", fontsize=10)

        # label ALL subplots with tokens (both axes)
        xt = _tick_positions(len(tokens), every_x)
        yt = _tick_positions(len(tokens[:M.shape[0]]), every_y)
        ax.set_xticks(xt)
        ax.set_xticklabels([tokens[i] for i in xt], rotation=90, fontsize=tick_fs)
        ax.set_yticks(yt)
        ax.set_yticklabels([tokens[i] for i in yt], fontsize=tick_fs)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    fig.suptitle(title_prefix, fontsize=13)
    # one shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label="Attention weight")
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=660, bbox_inches="tight")
    plt.show()

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(TOKENS_PATH, BATCH_INDEX)
    A = load_attention(ATTN_PATH)
    head_mats = extract_heads(A, BATCH_INDEX)  # list[(T,S)]

    if HEADS_TO_SHOW != "all":
        head_mats = [head_mats[h] for h in HEADS_TO_SHOW if h < len(head_mats)]

    # 1) trim at first EOS to drop trailing pads
    tokens_t, head_mats_t = trim_to_first_eos(tokens, head_mats, eos=EOS_TOKEN, keep_eos=True if not REMOVE_EOS else False)

    # 2) remove specials
    specials_cols = {CLS_TOKEN, PAD_TOKEN} | ({EOS_TOKEN} if (EOS_TOKEN and REMOVE_EOS) else set())
    specials_rows = {CLS_TOKEN, PAD_TOKEN} | ({EOS_TOKEN} if (EOS_TOKEN and REMOVE_EOS) else set())
    tokens_m, head_mats_m = mask_specials(tokens_t, head_mats_t, specials_cols, specials_rows)

    # 3) renormalize rows after masking
    if RENORMALIZE_ROWS:
        head_mats_m = [renorm_rows(M) for M in head_mats_m]

    title = f"{Path(ATTN_PATH).stem} — per-head attention (specials masked)"
    out_png = str(Path(OUT_DIR) / f"{Path(ATTN_PATH).stem}_perhead_masked_allticks.png")
    plot_heads_grid_all_ticks(
        head_mats_m, tokens_m, title, out_png,
        ncols=N_COLS, every_x=LABEL_EVERY_X, every_y=LABEL_EVERY_Y, tick_fs=TICK_FONTSIZE
    )

    # Generate 2D structure image for the SMILES
    smiles = "".join(tokens_m)  # Combine tokens to form the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img_path = str(Path(OUT_DIR) / f"{Path(ATTN_PATH).stem}_2d_structure.png")
        img = Draw.MolToImage(mol, size=(1200, 1200), legend=smiles)  # Add SMILES as legend
        img.save(img_path, dpi=(660, 660))
        print(f"2D structure image saved to {img_path}")
        print(f"SMILES: {smiles}")
    else:
        print("Failed to generate 2D structure image. Invalid SMILES.")

if __name__ == "__main__":
    main()
