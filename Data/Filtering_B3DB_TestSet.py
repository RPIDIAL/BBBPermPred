import pandas as pd
import numpy as np 
import RDKit
from RDKit import Chem

def filter_b3db_external(bbbp_path, b3db_path, output_path,sep_b3db="\t"):
    # Load the BBBP dataset (assumed to be used for training)
    bbbp_df = pd.read_csv(bbbp_path)
    # Load the B3DB dataset (from which we'll remove overlaps)
    b3db_df = pd.read_csv(b3db_path, sep=sep_b3db)
    
    # Normalize SMILES strings (strip whitespace and make lower-case if needed)
    # You may also want to canonicalize SMILES using RDKit if they are not in a canonical form.
    bbbp_df["smiles"] = bbbp_df["smiles"].astype(str).str.strip()
    b3db_df["smiles"] = b3db_df["SMILES"].astype(str).str.strip()
    
    # Create a set of SMILES from BBBP for fast lookup
    bbbp_smiles_set = set(bbbp_df["smiles"])
    
    # Filter out molecules in B3DB that are also in BBBP
    external_test_df = b3db_df[~b3db_df["SMILES"].isin(bbbp_smiles_set)].copy()
    
    print("Number of molecules in B3DB (original):", len(b3db_df))
    print("Number of molecules in external test set (B3DB minus BBBP):", len(external_test_df))
    
    # Save the external test set as a new CSV/TSV file
    external_test_df.to_csv(output_path, sep=sep_b3db, index=False)
    print(f"External test set saved to: {output_path}")


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

def filter_b3db_by_similarity(bbbp_path: str, b3db_path: str, output_path: str, threshold: float = 0.9, sep_b3db="\t"):
    # Load BBBP dataset (assumed to be used for training)
    bbbp_df = pd.read_csv(bbbp_path)
    # Load B3DB dataset (potential external test set)
    b3db_df = pd.read_csv(b3db_path, sep=sep_b3db)
    
    # Canonicalize SMILES in BBBP
    bbbp_df["canonical_smiles"] = bbbp_df["smiles"].astype(str).apply(get_canonical_smiles)
    # Remove rows that failed to canonicalize
    bbbp_df = bbbp_df.dropna(subset=["canonical_smiles"])
    # Compute fingerprints for BBBP
    bbbp_df["fingerprint"] = bbbp_df["canonical_smiles"].apply(get_fingerprint)
    bbbp_df = bbbp_df.dropna(subset=["fingerprint"])
    
    # Create a list of fingerprints from BBBP for fast lookup
    bbbp_fps = list(bbbp_df["fingerprint"])
    
    # Process B3DB: canonicalize and compute fingerprints
    b3db_df["canonical_smiles"] = b3db_df["SMILES"].astype(str).apply(get_canonical_smiles)
    b3db_df = b3db_df.dropna(subset=["canonical_smiles"])
    b3db_df["fingerprint"] = b3db_df["canonical_smiles"].apply(get_fingerprint)
    b3db_df = b3db_df.dropna(subset=["fingerprint"])
    
    # For each molecule in B3DB, determine if it is too similar to any BBBP molecule.
    b3db_df["is_similar"] = b3db_df["fingerprint"].apply(lambda fp: is_similar(fp, bbbp_fps, threshold))
    
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


if __name__ == "__main__":
    # Adjust these paths as needed.
    bbbp_dataset_path = "MolNet/BBBP.csv"
    b3db_dataset_path = "B3DB/B3DB/B3DB_classification.tsv"
    output_filtered_path =  "B3DB/B3DB/B3DBsim_external_test.tsv"
    
    # Set a Tanimoto similarity threshold (e.g., 0.9 means very similar)
    similarity_threshold = 0.9
    
    filter_b3db_by_similarity(
        bbbp_path=bbbp_dataset_path,
        b3db_path=b3db_dataset_path,
        output_path=output_filtered_path,
        threshold=similarity_threshold,
        sep_b3db="\t"
    )

