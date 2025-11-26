from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import os
from npyLoader import UrbanSoundNPYLoader

class Train:
    def __init__(self,
                 dataset_root_path,
                 name : str,
                 model, 
                 num_classes=10, 
                 batch_size=32, 
                 epochs=30, 
                 learning_rate=1e-3, 
                 patience=5, 
                 save_dir=os.path.join(os.getcwd(), "results")):
        

        self.dataset_root_path = dataset_root_path
        if not os.path.exists(dataset_root_path):
            raise ValueError(f"Dataset root path {dataset_root_path} does not exist.")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        
        self.name = name
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.early_stopping = True if patience > 0 else False
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.regularization = 1e-4  # L2 regularization strength
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print training configuration
        print("Training configuration:")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model.__class__.__name__}")

        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Parameters: total={total_params:,}, trainable={trainable_params:,}")
        except Exception:
            pass

        print(f"  Num classes: {self.num_classes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"  Save directory: {self.save_dir}")

        
    def epoch(self, dataloader):
        self.model.to(self.device)
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: inputs.shape={inputs.shape}, labels.shape={labels.shape}")
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            print(f"Batch {batch_idx}: outputs.shape={outputs.shape}")
            loss = self.criterion(outputs, labels)
            print(f"Batch {batch_idx}: loss={loss.item()}")
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy


    def validate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def run(self, folds):

        import copy
        print("\nStarting training with cross-validation...\n")

        history_per_fold = {}
        metrics_per_fold = {}

        # Salva o modelo inicial para resetar a cada fold
        initial_model_state = copy.deepcopy(self.model.state_dict())

        for test_idx, self.test_fold in enumerate(folds):
            # Reset do modelo para o estado inicial
            self.model.load_state_dict(initial_model_state)

            self.val_fold = folds[(test_idx + 1) % len(folds)]
            self.train_folds = [f for f in folds if f != self.test_fold or f != self.val_fold]
            print(f"\n=== Test = {self.test_fold}, Validation = {self.val_fold} ===")

            fold_history = {
                'train_acc': [], 
                'train_loss': [],
                'val_acc': [], 
                'val_loss': [],
                'test_acc': []
                }

            best_model_wts = None

            best_val_loss = float('inf')
            epochs_no_improve = 0

            from torch.utils.data import DataLoader
            test_loader = DataLoader(UrbanSoundNPYLoader(base_path=self.dataset_root_path, folders=[self.test_fold], pad_length=88200, shuffle=False), batch_size=self.batch_size, shuffle=False)
            val_loader = DataLoader(UrbanSoundNPYLoader(base_path=self.dataset_root_path, folders=[self.val_fold], pad_length=88200, shuffle=True), batch_size=self.batch_size, shuffle=False)
            train_loader = DataLoader(UrbanSoundNPYLoader(base_path=self.dataset_root_path, folders=self.train_folds, pad_length=88200, shuffle=True), batch_size=self.batch_size, shuffle=True)

            for epoch in range(1, self.epochs+1):
                print(f"\n--- Epoch {epoch}/{self.epochs} ---")
                training_acc, training_loss = self.epoch(train_loader)
                val_acc, val_loss = self.validate(val_loader)
                fold_history['train_acc'].append(training_acc)
                fold_history['train_loss'].append(training_loss)
                fold_history['val_acc'].append(val_acc)
                fold_history['val_loss'].append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_model_wts = self.model.state_dict()
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            print(f"Early stopping triggered after {epoch} epochs.")
                            break

                print(f"Train Loss: {training_loss:.4f}, Train Acc: {training_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            history_per_fold[self.test_fold] = fold_history 

            self.plot_history(fold_history)
            self.test_models(best_model_wts, self.model.state_dict(), test_loader) 
            self.save_json_metrics(history_per_fold, metrics_per_fold, save_path=os.path.join(self.save_dir, self.name, self.test_fold))

        self.save_json_metrics(history_per_fold, metrics_per_fold, save_path=os.path.join(self.save_dir, self.name, save_path=os.path.join(self.save_dir, self.name)))

    def test_models(self, best_model_wts, current_model_wts, dataloader, show=False):
        import matplotlib.pyplot as plt

        for model_name, model_wts in [("best", best_model_wts), ("current", current_model_wts)]:
            self.model.load_state_dict(model_wts)
            self.model.to(self.device)
            self.model.eval()


            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = 100 * correct / total if total > 0 else 0.0
            print(f"Test Accuracy of the {model_name} model on fold {self.test_fold}: {accuracy:.2f}%")

            # Confusion Matrix
            if all_labels and all_preds:
                cm = confusion_matrix(all_labels, all_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f"Confusion Matrix - {self.test_fold} ({model_name})")

                save_path = os.path.join(self.save_dir, self.name, self.test_fold)
                os.makedirs(save_path, exist_ok=True)

                plt.savefig(os.path.join(save_path, f"confusion_matrix_{model_name}.png"))
                if show:
                    plt.show()
                plt.close()
                print(f"Confusion matrix saved to {save_path}")
            else:
                print("No predictions or labels to plot confusion matrix.")

    def plot_history(self, history, show=False):
        import matplotlib.pyplot as plt

        os.makedirs(self.save_dir, exist_ok=True)

        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, self.name, self.test_fold)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        plt.savefig(os.path.join(save_path, f"training_history.png"))

        print(f"Saved training history plot to {save_path}")

        if show:
            plt.show()

        plt.close()

    def save_json_metrics(self, history_per_fold, metrics_per_fold, save_path=None):
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save history per fold
        history_path = os.path.join(save_path, "history_per_fold.json")
        with open(history_path, 'w') as f:
            json.dump(history_per_fold, f, indent=4)
        print(f"Saved history per fold to {history_path}")

        # Save metrics per fold
        metrics_path = os.path.join(save_path, "metrics_per_fold.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_per_fold, f, indent=4)
        print(f"Saved metrics per fold to {metrics_path}")




# ============== MOCK TEST =================
from models.CNN import SoundCNN

import os

FOLDS = [f"fold{i}" for i in range(1, 11)]
DATA_PATH = "datasets/augmentation"

# Pass an already instantiated model
model_instance = SoundCNN(num_classes=10, SqueezeExcitation=False)
trainer = Train(dataset_root_path=DATA_PATH, name="CNN", model=model_instance, num_classes=10, batch_size=128, epochs=1, learning_rate=1e-3, patience=5, save_dir="results")
trainer.run(folds=FOLDS)