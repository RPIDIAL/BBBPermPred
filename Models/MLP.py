import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import os

# currently copy-pasted from gpt must be fixed 
class MLPClassifier(nn.Module):

    def __init__(
            self, 
            input_dim,
            batch_size,  
            learning_rate,
            dropout=0.1,
            n_epochs=100,
            ):
        
        super(MLPClassifier, self).__init__()


        self.model =  nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.loss = nn.BCELoss()
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def get_embeddings(self, X, y, device='cpu'):
        self.to(device)
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            # get output from the second last layer
            embeddings = self.model[:-2](inputs).cpu().numpy()
            print(f"Embeddings Shape using [:-2]: {embeddings.shape}")
            print(f"Embeddings Shape using [:-1]: {self.model[:-1]}: {embeddings.shape}")

        return embeddings, y


    def forward(self, x):
        return self.model(x)


    def fit(
            self, 
            X_train, 
            y_train, 
            X_val,
            y_val,
            device='cpu'
        ):
        self.to(device)

        lr = self.learning_rate
        batch_size = self.batch_size
        epochs = self.n_epochs

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.train()

        best_checkpoints = []

        for epoch in range(epochs):

            train_epoch_loss = 0.0
            val_epoch_loss = 0.0

            #train step
            for features, labels in train_dataloader:
                features, labels = features.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = self(features)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()

            average_train_loss = train_epoch_loss / len(train_dataloader)

            #validation step
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device).unsqueeze(1)

                with torch.no_grad():
                    outputs = self(features)
                    val_epoch_loss = criterion(outputs, labels)
                    val_epoch_loss += val_epoch_loss.item()

            average_val_loss = val_epoch_loss / len(val_dataloader)
                
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}")


            # Save current checkpoint with filename that includes epoch and validation loss
            checkpoint_filename = f'checkpoint_epoch_{epoch + 1:02d}_valloss_{average_val_loss:.4f}.pt'
            self.save_checkpoint(checkpoint_filename)
            best_checkpoints.append((average_val_loss, checkpoint_filename))
            # Sort checkpoints by loss (ascending order; lower is better)
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0])
            # Remove checkpoints if we have more than the top 5
            if len(best_checkpoints) > 5:
                # Remove worst performing checkpoint (last in sorted list)
                worst_loss, worst_file = best_checkpoints.pop()
                if os.path.exists(worst_file):
                    os.remove(worst_file)


        print(f"Training complete. Best checkpoints saved.{[f' {loss:.4f} ({file})' for loss, file in best_checkpoints]}")


    def predict(self, X, device='cpu', threshold=0.5):
        
        self.to(device)
        self.eval()

        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            probs = self(inputs).cpu().numpy()

        preds = (probs > threshold).astype(int)
        return preds.flatten(), probs.flatten()
    

    def evaluate(self, X, y, device='cpu', threshold=0.5):
        preds, probs = self.predict(X, device, threshold)

        accuracy = accuracy_score(y, preds)
        roc_auc = roc_auc_score(y, probs)
        pr_auc = average_precision_score(y, probs)

        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        }

        print(f"Evaluation Metrics:\n Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        return metrics, probs
    
    def predict_proba(self, X):
        """
        Returns an n×2 array of class probabilities, as scikit-learn expects.
        """
        device = 'cpu'
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            p1 = self(X_tensor).cpu().numpy().flatten()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


    def log_train_metrics():
        pass
    
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

