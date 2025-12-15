#The purpose of this file is to save everything to a csv IN A FOLDER SO ITS ORGANIZED 
import hydra
import pandas as pd
import torch
import numpy as np
import os
import time

import Chemformer.molbart.utils.data_utils as util
from Models.ChemformerClassifier import ChemformerClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from rdkit import Chem
from deepchem.feat import RDKitDescriptors


t_on = "B3DB" # training set

@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(args):

    datasets = ["NeuTox", "BBBP", "B3DB"]

    splits = ["train", "test", "val", "full"]

    t0 = time.time()

    k_folds = 5
    for fold in range(k_folds):
        print(f"Processing fold {fold + 1}/{k_folds}")
        
        for dataset in datasets:

            if dataset == "NeuTox":
                args.data_path = "Data/NeuTox_filtered.csv"
                args.datamodule = ["molbart.data.base.NeuToxDataModule"]
            elif dataset == "BBBP":
                args.data_path = "Data/MolNet/BBBP.csv"
                args.datamodule = ["molbart.data.base.BBBPDataModule"]
            elif dataset == "B3DB":
                args.data_path = "Data/B3DB/B3DB/B3DBsim_external_test.tsv"
                args.datamodule = ["molbart.data.base.B3DBDataModule"]
            else:
                raise ValueError("Invalid dataset")


            for split in splits:
                args.dataset_part = split

                chemformer = ChemformerClassifier(args)    
                chemformer.model.eval()

                output_folder = f"Output_Files/{t_on}/Avg_Pool/NoLayerNorm/{dataset}/{split}/{fold}"
                os.makedirs(output_folder, exist_ok=True)

                embs, labels, descs, comb = chemformer.get_combined_features(dataset_part=split, 
                                                                             layernorm=False, 
                                                                             preconcat_layernorm=False,
                                                                             cls=False,
                                                                             avg_pooling=True,
                                                                             comb=False)                
                
                embs = pd.DataFrame(embs)
                embs['label'] = labels
                embs.to_csv(f"{output_folder}/Embeddings.csv", index=False)
                descs = pd.DataFrame(descs)
                descs['label'] = labels
                descs.to_csv(f"{output_folder}/Descriptors.csv", index=False)
                comb = pd.DataFrame(comb)
                comb['label'] = labels
                comb.to_csv(f"{output_folder}/Combined.csv", index=False)

                print(f"Files saved for {dataset} {split} set")
                print("Shape of embs: ", embs.shape)    
                print("Shape of descs: ", descs.shape)
                print("Shape of comb: ", comb.shape)

                del chemformer
    
    tf = time.time()
    total_time = tf - t0
    print("Total time taken: ", total_time/60, " minutes")

if __name__ == "__main__":
    main()