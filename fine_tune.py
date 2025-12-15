import time

import hydra

import Chemformer.molbart.utils.data_utils as util
from Models.ChemformerClassifier import ChemformerClassifier
from Models.ClassifierModel import ClassifierModel


@hydra.main(version_base=None, config_path="config", config_name="fine_tune")
def main(args):

    util.seed_everything(args.seed)

    print("Fine-tuning CHEMFORMER.")
    t0 = time.time()
    chemformer = ChemformerClassifier(args)
  
    chemformer.fit()
    t_fit = time.time() - t0
    print(f"Training complete, time: {t_fit}")
    print("Done fine-tuning.")
    print("Running a test step...")
    chemformer.trainer.test() 

    return


if __name__ == "__main__":
    main()
