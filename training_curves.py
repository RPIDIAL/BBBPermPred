import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

b3db = "Chemformer/FineTune_Curves/B3DB/Avg_Pool/classification/version_0/logged_train_metrics.csv"

b3db = pd.read_csv(b3db, sep = "\t")
print("the columns", b3db.columns.tolist())


val_loss = b3db["val_loss"]
val_acc = b3db["val_acc"]
val_roc_auc = b3db["val_roc_auc"]
epoch = b3db["epoch"]
train_loss_epoch = b3db["train_loss_epoch"]
plt.plot(epoch, val_loss, label="Val Loss", alpha=0.5)
plt.plot(epoch, val_acc, label="Val Accuracy")
plt.plot(epoch, b3db["val_roc_auc"], label="Val ROC AUC")
plt.plot(epoch, train_loss_epoch, label="Train Loss", alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss/Accuracy")
plt.title("B3DB Fine-Tuning")
plt.show()
plt.savefig("Chemformer/Sept2025_FineTune/B3DB_FT_400epoch.png", dpi=660)
plt.close()


neutox = "Chemformer/FineTune_Curves/NeuTox/Linear_Schedule/No_Freezing/BCE_loss/CLS_Token/classification/version_1/logged_train_metrics.csv"
neutox = pd.read_csv(neutox, sep = "\t")
print("the columns", neutox.columns.tolist())
val_loss = neutox["val_loss"]
val_acc = neutox["val_acc"]
val_roc_auc = neutox["val_roc_auc"]
epoch = neutox["epoch"]
train_loss_epoch = neutox["train_loss_epoch"]
plt.plot(epoch, val_loss, label="Val Loss", alpha = 0.5)
plt.plot(epoch, val_acc, label="Val Accuracy")
plt.plot(epoch, neutox["val_roc_auc"], label="Val AUROC")
plt.plot(epoch, train_loss_epoch, label="Train Loss", alpha = 0.5)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss/Accuracy")
plt.title("NeuTox Fine-Tuning")
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 1, 0, 3]  # Order: Val AUROC, Val Accuracy, Val Loss, Train Loss
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.show()
plt.savefig("Chemformer/Sept2025_FineTune/NeuTox_FT_400epoch.png", dpi=660)
plt.close()

