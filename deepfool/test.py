import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.CNN import SoundCNN
import torch
from dataloader import Dataloader
from preprocessing import AudioPreprocessor
from sklearn.metrics import confusion_matrix
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

single_model = SoundCNN(num_classes=10, SqueezeExcitation=False, in_channels=1)
single_model_path = r"C:\deep-learning-urban-sound-data\models_for_adversarial\CNN_singlechannel_all_folds.pth"
single_model.load_state_dict(torch.load(single_model_path, map_location=device))
single_model.to(device)
single_model.eval()


multi_model = SoundCNN(num_classes=10, SqueezeExcitation=True, AttentionBlock=True, in_channels=5)
multi_model_path = r"C:\deep-learning-urban-sound-data\models_for_adversarial\ASECNN_all_folds.pth"
multi_model.load_state_dict(torch.load(multi_model_path, map_location=device))
multi_model.to(device)
multi_model.eval()

FOLDS_TO_TEST = ["fold9", "fold10"]
PATH = r"C:\deep-learning-urban-sound-data\datasets"
preprocessor = AudioPreprocessor()
dataloader = Dataloader(dataset_path=PATH,
                        folds=FOLDS_TO_TEST, use_cache=True, preprocessing=preprocessor)

single_preds = []
multi_preds = []
true_labels = []

with torch.no_grad():
    for i in range(len(dataloader)):
        inputs, fold, labels = dataloader[i]
        inputs = torch.from_numpy(inputs).unsqueeze(0).float().to(device)
        
        # Single channel model prediction
        # Assuming single channel model takes the first channel or a specific slice
        # Adjust input shape if necessary based on model expectation, here assuming [B, 1, H, W]
        single_input = inputs[:, 0:1, :, :] 
        outputs_single = single_model(single_input)
        _, predicted_single = torch.max(outputs_single.data, 1)
        single_preds.extend(predicted_single.cpu().numpy())

        # Multi channel model prediction
        outputs_multi = multi_model(inputs)
        _, predicted_multi = torch.max(outputs_multi.data, 1)
        multi_preds.extend(predicted_multi.cpu().numpy())

        true_labels.append(labels)

cm_single = confusion_matrix(true_labels, single_preds)
cm_multi = confusion_matrix(true_labels, multi_preds)

print("Confusion Matrix for Single Channel Model:")
print(cm_single)
print("\nConfusion Matrix for Multi Channel Model:")
print(cm_multi)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(cm_single, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Single Channel Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - Multi Channel Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()