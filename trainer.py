from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import os
from dataloader import Dataloader as MyLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import copy
import json

class Train:
    def __init__(self,
                 dataset_root_path,
                 name : str,
                 model,
                 dataset_type: str = "preprocessing",
                 num_classes=10, 
                 batch_size=32, 
                 epochs=30, 
                 learning_rate=1e-3, 
                 patience=5, 
                 save_dir=os.path.join(os.getcwd(), "results")):
        

        assert dataset_type in ["preprocessing", "augmentation_preprocessing", "singlechannel", "augmentation_singlechannel"], f"Invalid dataset_type: {dataset_type}. Must be one of ['preprocessing', 'augmentation_preprocessing', 'singlechannel', 'augmentation_singlechannel']."
        self.dataset_type = dataset_type

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
        print(f"  Dataset type: {self.dataset_type}")
        print("Loss function: CrossEntropyLoss")
        print(f"Regularization (L2 weight decay): {self.regularization}")
        print(f"Optimizer: Adam\n")

        
    def epoch(self, dataloader):
        
        self.model.to(self.device)
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=True)
        for batch_idx, (inputs, folds, labels) in enumerate(progress_bar):

            #print(f"Inputs shape: {inputs.shape}")

            # Seleciona apenas o primeiro canal de cada input
            if self.dataset_type == "singlechannel" or self.dataset_type == "augmentation_singlechannel":
                inputs = inputs[:, 0, ...].unsqueeze(1)

            #print(f"Inputs shape after channel selection: {inputs.shape}")

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Atualizar barra de progresso
            current_acc = correct / total if total > 0 else 0.0
            current_loss = running_loss / total if total > 0 else 0.0
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
            
        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate(self, dataloader):
        
        self.model.to(self.device)
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(dataloader, desc="Validating", leave=True)
        with torch.no_grad():
            for inputs, folds, labels in progress_bar:

                if self.dataset_type == "singlechannel" or self.dataset_type == "augmentation_singlechannel":
                    inputs = inputs[:, 0, ...].unsqueeze(1)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Atualizar barra de progresso
                current_acc = correct / total if total > 0 else 0.0
                current_loss = running_loss / total if total > 0 else 0.0
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
                
        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def run(self, folds):

        
        print("\nStarting training with cross-validation...\n")

        history_per_fold = {}
        metrics_per_fold = {}

        # Salva o modelo inicial para resetar a cada fold
        initial_model_state = copy.deepcopy(self.model.state_dict())

        for test_idx, self.test_fold in enumerate(folds):
            # Reset do modelo para o estado inicial
            self.model.load_state_dict(copy.deepcopy(initial_model_state))

            self.val_fold = folds[(test_idx + 1) % len(folds)]
            self.train_folds = [f for f in folds if f != self.test_fold and f != self.val_fold]
            print(f"\n=== Train {self.train_folds}, Validation = {self.val_fold}, Test = {self.test_fold} ===")

            fold_history = {
                'train_acc': [], 
                'train_loss': [],
                'val_acc': [], 
                'val_loss': []
                }

            best_model_wts = None

            best_val_loss = float('inf')
            epochs_no_improve = 0

            
            if self.dataset_type == "preprocessing" or self.dataset_type == "singlechannel":
                test_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=[self.test_fold], include_augmented=False, use_cache=True), batch_size=self.batch_size, shuffle=False)
                val_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=[self.val_fold], include_augmented=False, use_cache=True), batch_size=self.batch_size, shuffle=False)
                train_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=self.train_folds, include_augmented=False, use_cache=True), batch_size=self.batch_size, shuffle=True)
            else:  # augmentation_preprocessing or augmentation_singlechannel
                test_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=[self.test_fold], include_augmented=True, use_cache=True), batch_size=self.batch_size, shuffle=False)
                val_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=[self.val_fold], include_augmented=True, use_cache=True), batch_size=self.batch_size, shuffle=False)
                train_loader = DataLoader(MyLoader(dataset_path=self.dataset_root_path, folds=self.train_folds, include_augmented=True, use_cache=True), batch_size=self.batch_size, shuffle=True)

            for epoch in range(1, self.epochs+1):
                print(f"\n--- Epoch {epoch}/{self.epochs} ---")
                training_loss, training_acc = self.epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)
                fold_history['train_acc'].append(training_acc)
                fold_history['train_loss'].append(training_loss)
                fold_history['val_acc'].append(val_acc)
                fold_history['val_loss'].append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            print(f"Early stopping triggered after {epoch} epochs.")
                            break

                print(f"Train Loss: {training_loss:.4f}, Train Acc: {training_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Calcular métricas no test set
            if best_model_wts is not None:
                metrics = self.test_models(best_model_wts, test_loader, self.test_fold)
            else:
                print("Aviso: Nenhum melhor modelo foi salvo.")
                metrics = {}

            # Guardar histórico e métricas
            history_per_fold[self.test_fold] = fold_history 
            metrics_per_fold[self.test_fold] = metrics

            self.plot_history(fold_history)
            
            # Guardar resultados específicos deste fold
            fold_save_path = os.path.join(self.save_dir, self.dataset_type, self.name, self.test_fold)
            self.save_fold_results(fold_history, metrics, fold_save_path)

        # Guardar resumo geral de todos os folds
        self.save_json_metrics(history_per_fold, metrics_per_fold, save_path=os.path.join(self.save_dir, self.dataset_type, self.name, "overall"))

    def test_models(self, best_model_wts, dataloader, test_fold_name, show=False):
        

        self.model.load_state_dict(best_model_wts)
        self.model.to(self.device)
        self.model.eval()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(dataloader, desc="Testing", leave=True)
        with torch.no_grad():
            for inputs, folds, labels in progress_bar:
                if self.dataset_type == "singlechannel" or self.dataset_type == "augmentation_singlechannel":
                    inputs = inputs[:, 0, ...].unsqueeze(1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Atualizar barra de progresso
                current_acc = correct / total if total > 0 else 0.0
                progress_bar.set_postfix({'acc': f'{current_acc:.4f}'})

        accuracy = correct / total if total > 0 else 0.0
        print(f"Test Accuracy on fold {test_fold_name}: {accuracy:.4f}")

        # Confusion Matrix
        if all_labels and all_preds:
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {test_fold_name}")

            save_path = os.path.join(self.save_dir, self.dataset_type, self.name, test_fold_name)
            os.makedirs(save_path, exist_ok=True)

            plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
            if show:
                plt.show()
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
        else:
            print("No predictions or labels to plot confusion matrix.")

        # Métricas adicionais
        accuracy_per_class = (cm.diagonal() / cm.sum(axis=1)).tolist() if total > 0 and len(cm) > 0 else None
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        # G-means: sqrt(prod(recall_per_class))
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        G_means = float(np.prod(recall_per_class) ** (1.0 / len(recall_per_class))) if len(recall_per_class) > 0 else 0.0

        metrics = {
            'accuracy': float(accuracy),
            'accuracy_per_class': accuracy_per_class,
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'G_means': float(G_means)
        }

        return metrics

    def plot_history(self, history, show=False):
        

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

        save_path = os.path.join(self.save_dir, self.dataset_type, self.name, self.test_fold)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "training_history.png"))
        print(f"Saved training history plot to {save_path}")
        if show:
            plt.show()
        plt.close()

    def save_fold_results(self, history, metrics, save_path):
        
        
        os.makedirs(save_path, exist_ok=True)

        # Save fold history
        history_path = os.path.join(save_path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved fold history to {history_path}")

        # Save fold metrics (test results)
        metrics_path = os.path.join(save_path, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved fold test metrics to {metrics_path}")

    def save_json_metrics(self, history, metrics, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.save_dir, self.dataset_type, self.name)

        os.makedirs(save_path, exist_ok=True)

        # Save overall history from all folds
        history_path = os.path.join(save_path, "all_folds_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved all folds history to {history_path}")

        # Save overall metrics from all folds
        metrics_path = os.path.join(save_path, "all_folds_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved all folds metrics to {metrics_path}")



'''
# ============== MOCK TEST =================
from models.CNN import SoundCNN

import os

FOLDS = [f"fold{i}" for i in range(1, 11)]
DATA_PATH = "datasets"

# Pass an already instantiated model
model_instance = SoundCNN(num_classes=10, SqueezeExcitation=False)
trainer = Train(dataset_root_path=DATA_PATH,name="CNN", dataset_type="singlechannel", model=model_instance, num_classes=10, batch_size=128, epochs=2, learning_rate=1e-3, patience=5)
trainer.run(folds=FOLDS)'''