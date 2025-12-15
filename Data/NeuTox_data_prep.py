# Prepping the NeuTox dataset 
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_canonical_smiles(smiles: str) -> str:
    """Convert a SMILES string to its canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_fingerprint(smiles: str):
    """Compute a Morgan fingerprint for a canonical SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def is_similar(fp, fps_list, threshold: float) -> bool:
    """
    Check if fingerprint fp has a Tanimoto similarity greater than or equal to
    threshold with any fingerprint in fps_list.
    """
    for ref_fp in fps_list:
        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
        if sim >= threshold:
            return True
    return False


NeuTox = "/MolNet/NeuTox.csv"
b3db = "/B3DB/B3DB_classification.tsv"
bbbp = "/MolNet/BBBP.csv"
b3db_df = pd.read_csv(b3db, sep="\t")
b3db_df["BBB+/BBB-"] = b3db_df["BBB+/BBB-"].map({"BBB+": 1, "BBB-": 0})
b3db_df["canonical_smiles"] = b3db_df["SMILES"].astype(str).apply(get_canonical_smiles)
b3db_df = b3db_df.dropna(subset=["canonical_smiles"])
b3db_df["fingerprint"] = b3db_df["canonical_smiles"].apply(get_fingerprint)
b3db_df = b3db_df.dropna(subset=["fingerprint"])
b3db_fps = list(b3db_df["fingerprint"])

NeuTox_df = pd.read_csv(NeuTox)
NeuTox_df["canonical_smiles"] = NeuTox_df["SMILES"].astype(str).apply(get_canonical_smiles)
NeuTox_df = NeuTox_df.dropna(subset=["canonical_smiles"])
NeuTox_df["fingerprint"] = NeuTox_df["canonical_smiles"].apply(get_fingerprint)
NeuTox_df = NeuTox_df.dropna(subset=["fingerprint"])


bbbp_df = pd.read_csv(bbbp)
bbbp_df["canonical_smiles"] = bbbp_df["smiles"].astype(str).apply(get_canonical_smiles)
bbbp_df = bbbp_df.dropna(subset=["canonical_smiles"])
bbbp_df["fingerprint"] = bbbp_df["canonical_smiles"].apply(get_fingerprint)
bbbp_df = bbbp_df.dropna(subset=["fingerprint"])
bbbp_fps = list(bbbp_df["fingerprint"])

threshold = 0.9

NeuTox_df["is_similar"] = NeuTox_df["fingerprint"].apply(lambda fp: is_similar(fp, bbbp_fps, threshold))
print("Original number of molecules in NeuTox:", len(NeuTox_df))

filtered_df = NeuTox_df[~NeuTox_df["is_similar"]].copy()
print("Number of molecules after filtering (Tanimoto >= {:.2f} removed):".format(threshold), len(filtered_df))

filtered_df = filtered_df.drop(columns=["is_similar"])

filtered_df["is_similar"] = filtered_df["fingerprint"].apply(lambda fp: is_similar(fp, bbbp_fps, threshold))

filtered_df = filtered_df[~filtered_df["is_similar"]].copy()

print("Number of molecules after filtering (Tanimoto >= {:.2f} removed):".format(threshold), len(filtered_df))

filtered_df = filtered_df.drop(columns=["is_similar", "canonical_smiles", "fingerprint"])

filtered_df.to_csv("NeuTox_filtered.csv", index=False)

"""
    # Filter out molecules from B3DB that are too similar to any BBBP molecule.
    filtered_df = b3db_df[~b3db_df["is_similar"]].copy()
    
    print("Original number of molecules in B3DB:", len(b3db_df))
    print("Number of molecules after filtering (Tanimoto >= {:.2f} removed):".format(threshold), len(filtered_df))
    
    # Optionally, drop intermediate columns before saving.
    filtered_df = filtered_df.drop(columns=["canonical_smiles", "fingerprint", "is_similar"])
    filtered_df.to_csv(output_path, sep=sep_b3db, index=False)
    print(f"Filtered external test set saved to: {output_path}")

    # Save the dropped molecules (those that were too similar) to a separate CSV/TSV file
    dropped_df = b3db_df[b3db_df["is_similar"]].copy()
    dropped_output_path = output_path.replace(".tsv", "_dropped.tsv").replace(".csv", "_dropped.csv")
    dropped_df = dropped_df.drop(columns=["canonical_smiles", "fingerprint", "is_similar"])
    dropped_df.to_csv(dropped_output_path, sep=sep_b3db, index=False)
    print(f"Dropped molecules saved to: {dropped_output_path}, and a total of {len(dropped_df)} molecules dropped.")
"""






