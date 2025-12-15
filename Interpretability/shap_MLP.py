import shap
import torch
import matplotlib.pyplot as plt
from BBBPermPred.Models.EnsembleClassifier import EnsembleClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
fold = 0
set_name = "B3DB"

t_on = "B3DB"


# train the ensemble model first on the training data [B3DB]
model = EnsembleClassifier(
                        batch_size=16,
                        learning_rate=1e-5,
                        n_epochs=150
                    )
base = f"/fast/grantn2/BBBPermeability/Chemformer/BCE_Outputs_250epochs/{t_on}/Avg_Pool/NoLayerNorm"

x_train_desc = pd.read_csv(f"{base}/{t_on}/train/{fold}/Descriptors.csv").drop(columns=["label"]).values
x_train_embed= pd.read_csv(f"{base}/{t_on}/train/{fold}/Embeddings.csv").drop(columns=["label"]).values
x_train_concat= pd.read_csv(f"{base}/{t_on}/train/{fold}/Combined.csv").drop(columns=["label"]).values

x_val_desc   = pd.read_csv(f"{base}/{t_on}/val/{fold}/Descriptors.csv").drop(columns=["label"]).values
x_val_embed  = pd.read_csv(f"{base}/{t_on}/val/{fold}/Embeddings.csv").drop(columns=["label"]).values
x_val_concat = pd.read_csv(f"{base}/{t_on}/val/{fold}/Combined.csv").drop(columns=["label"]).values

x_test_desc  = pd.read_csv(f"{base}/{set_name}/test/{fold}/Descriptors.csv").drop(columns=["label"]).values
x_test_embed = pd.read_csv(f"{base}/{set_name}/test/{fold}/Embeddings.csv").drop(columns=["label"]).values
x_test_concat= pd.read_csv(f"{base}/{set_name}/test/{fold}/Combined.csv").drop(columns=["label"]).values

y_train      = pd.read_csv(f"{base}/{t_on}/train/{fold}/Descriptors.csv")["label"].values
y_val        = pd.read_csv(f"{base}/{t_on}/val/{fold}/Descriptors.csv")["label"].values
y_test       = pd.read_csv(f"{base}/{set_name}/test/{fold}/Descriptors.csv")["label"].values

scaler_desc = StandardScaler()
scaler_embed = StandardScaler()
scaler_concat = StandardScaler()

x_train_desc = scaler_desc.fit_transform(x_train_desc)
x_train_embed = scaler_embed.fit_transform(x_train_embed)
x_train_concat = scaler_concat.fit_transform(x_train_concat)

x_val_desc = scaler_desc.transform(x_val_desc)
x_val_embed = scaler_embed.transform(x_val_embed)
x_val_concat = scaler_concat.transform(x_val_concat)

x_test_desc = scaler_desc.transform(x_test_desc)
x_test_embed = scaler_embed.transform(x_test_embed)
x_test_concat = scaler_concat.transform(x_test_concat)



# Get that names of the RDKit Features
desc_feature_names = pd.read_csv("descriptor_name_dict.csv")
desc_feature_names = desc_feature_names.iloc[:, 1].tolist()



model.fit(
    X_desc=x_train_desc,
    X_embed=x_train_embed,
    X_concat=x_train_concat,
    y=y_train,
    X_val_desc=x_val_desc,
    X_val_embed=x_val_embed,
    X_val_concat=x_val_concat,
    y_val=y_val,
    device='cpu'
)

# want SHAP just for the Descriptor_Head
desc_model = model.Descriptor_Head
desc_model.eval()

# pick a background set from your descriptor data
bg_size = min(200, len(x_train_desc))
background = torch.tensor(x_train_desc[:bg_size], dtype=torch.float32)

# DeepExplainer works directly with PyTorch models
explainer = shap.DeepExplainer(desc_model, background)

# subset test set for explanation (avoid huge arrays)
n_explain = min(256, len(x_test_desc))
Xt = torch.tensor(x_test_desc[:n_explain], dtype=torch.float32)

# get shap values (probability output since model ends with sigmoid)
shap_values = explainer.shap_values(Xt)

# convert to numpy if needed
if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

X_slice = x_test_desc[:n_explain]


out_dir = "Attention_Visualization/shap"

# summary plot (beeswarm)
shap.summary_plot(shap_vals, X_slice, feature_names=desc_feature_names, show=False)
plt.tight_layout(); plt.show()
plt.savefig(os.path.join(out_dir, f"shap_summary_{t_on}_fold{fold}_{set_name}.png"), bbox_inches='tight', dpi=660)

# bar plot (global feature importance)
shap.summary_plot(shap_vals, X_slice, feature_names=desc_feature_names,
                  plot_type="bar", show=False)
plt.tight_layout(); plt.show()
plt.savefig(os.path.join(out_dir, f"shap_bar_{t_on}_fold{fold}_{set_name}.png"), bbox_inches='tight', dpi=660)
