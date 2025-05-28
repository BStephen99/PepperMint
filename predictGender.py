import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import glob
import torch.nn.functional as F

class GenderDataset(Dataset):
    def __init__(self, pkl_files):
        self.embeddings = []
        self.labels = []
        self.genders = []
        self.persons = []
        self.frame_keys = []

        for file in pkl_files:
            print(file)
            with open(file, 'rb') as f:
                data = pickle.load(f)
                for frame_key, persons in data.items():
                    notalker = all(label == 0 for label in [person["label"] for person in persons])
                    if notalker:
                        emb = persons[0]['speakerEmb']
                        gender = "noTalk"
                        personID = persons[0]['person_id']
                        if emb is not None and len(emb) == 192:
                                self.embeddings.append(emb)
                                self.genders.append(gender)
                                self.persons.append(personID)
                                self.frame_keys.append(frame_key)
                    for person in persons:
                        label = person['label'][0]
                        if label in [1, 2]:  # only if speaking
                            emb = person['speakerEmb']
                            gender = person['gender']
                            personID = person['person_id']
                            #if "pepper" in personID:
                            #    continue
                            #print(gender)
                            if emb is not None and len(emb) == 192:
                                self.embeddings.append(emb)
                                self.genders.append(gender)
                                self.persons.append(personID)
                                self.frame_keys.append(frame_key)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.genders)

    def __len__(self):
        return len(self.embeddings)


    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        person_id = self.persons[idx]
        frame_key = self.frame_keys[idx]
        return emb, label, person_id, frame_key


class GenderClassifier(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_classes=4):
        super(GenderClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_loader , num_epochs=10, lr=1e-3):
    #train_data, test_data = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=42)


    model = GenderClassifier()
    #model.load_state_dict(torch.load('gender_model.pt'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y, person_id, frame_key in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")
        evaluate_model(model, test_loader, encoder)

    #return model, test_loader, dataset.label_encoder
    return model

from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y, person_id, frame_key in test_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.tolist())
            all_labels.extend(y.tolist())

    acc = correct / total
    print(f"Test Accuracy: {acc:.2%}")

    # Precision, Recall, F1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(label_encoder.classes_)), zero_division=0
    )

    class_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    print("\nPer-Class Metrics:")
    for i, label in enumerate(class_labels):
        print(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}")



def get_most_confident_wrong_predictions(model, test_loader, label_encoder, top_k=5):
    model.eval()
    wrong_samples = []

    with torch.no_grad():
        for X, y, person_id, frame_key in test_loader:
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            for i in range(len(y)):
                if preds[i] != y[i]:
                    wrong_samples.append({
                        'input': X[i],
                        'true_label': y[i].item(),
                        'pred_label': preds[i].item(),
                        'person': person_id[i],
                        'frame_key': frame_key[i],
                        'confidence': confidences[i].item()
                    })

    # Sort by confidence descending and return top_k wrong predictions
    wrong_samples.sort(key=lambda x: x['confidence'], reverse=True)
    top_wrong = wrong_samples[:top_k]

    # Decode labels to human-readable
    for sample in top_wrong:
        sample['true_label_str'] = label_encoder.inverse_transform([sample['true_label']])[0]
        sample['pred_label_str'] = label_encoder.inverse_transform([sample['pred_label']])[0]

    print(top_wrong)

    return top_wrong




pkl_filesTrain =  glob.glob("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/train/*")
#pkl_filesTrain = glob.glob("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/WASDtrain/*")[:800] 
pkl_filesTest  =  glob.glob("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/test/*") 
batch_size=32



train = GenderDataset(pkl_filesTrain)
test = GenderDataset(pkl_filesTest)


from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Count the number of samples per class
labels = train.labels  # assuming train.labels is a list or tensor of label indices
class_sample_counts = np.bincount(labels)

# Inverse of class frequency for balancing
weights = 1. / class_sample_counts
sample_weights = [weights[label] for label in labels]

# Create a sampler
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)

# DataLoader with balanced sampling
train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler)




#train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size)
encoder = train.label_encoder


#model = train_model(train_loader, num_epochs=50, lr=1e-3)


#torch.save(model.state_dict(), "gender_model.pt")
#evaluate_model(model, test_loader, encoder)


#get_most_confident_wrong_predictions(model, test_loader, encoder, top_k=5)




import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np

def load_model_and_save_frame_predictions_plot(
    pkl_file, 
    model_path, 
    model_class, 
    label_encoder, 
    save_path='frame_gender_predictions.png'
):
    # Step 1: Load model
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Step 2: Load data
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    frame_gender_map = {}

    # Step 3: Make predictions for all people in all frames
    missing_emb_frames = []

    with torch.no_grad():
        for frame_key, persons in data.items():
            print(frame_key)
            if not persons:
                continue  # no people in this frame

            emb = persons[0].get('speakerEmb')  # assuming shared
            if emb is not None and len(emb) == 192:
                emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                logits = model(emb_tensor)
                probs = F.softmax(logits, dim=1)
                print(probs)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_label = label_encoder.inverse_transform([pred_class])[0]
                print(pred_label)
                frame_gender_map[frame_key] = pred_label
            else:
                missing_emb_frames.append(frame_key)

    print(f"Total frames: {len(data)}")
    print(f"Frames missing valid emb: {len(missing_emb_frames)}")
    print("Example missing frames:", missing_emb_frames[:5])

    # Step 4: Prepare for plotting
    all_frames = sorted(data.keys(), key=lambda x: float(x))
    categories = sorted(label_encoder.classes_)

    # Use distinct colors
    cmap = plt.get_cmap('Set1')
    color_map = {label: cmap(i) for i, label in enumerate(categories)}

    # Step 5: Plot each frame as a segment of a color-coded line
    fig, ax = plt.subplots(figsize=(12, 2))
    y_value = 0.5  # constant y

    for i in range(len(all_frames) - 1):
        frame = all_frames[i]
        #print(frame)
        next_frame = all_frames[i + 1]
        pred_label = frame_gender_map.get(frame, None)
        #print(pred_label)
        if pred_label is not None:
            ax.plot(
                [float(frame), float(next_frame)],
                [y_value, y_value],
                color=color_map[pred_label],
                linewidth=100
            )

    ax.set_yticks([])

    ax.set_xlabel("Frame Timestamp")
    ax.set_title("Predicted Gender per Frame")

    # Build a custom legend
    handles = [plt.Line2D([0], [0], color=color_map[label], lw=4, label=label) for label in categories]
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    #plt.xlim(float(all_frames[0]), float(all_frames[-1]))
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction plot to: {save_path}")


print("***************",pkl_filesTest[3])

load_model_and_save_frame_predictions_plot(
    pkl_file="/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/test/220927_CLIP_43.pkl", #pkl_filesTest[3],
    model_path='gender_model.pt',
    model_class=GenderClassifier,
    label_encoder=encoder,
    save_path='gender_predictions_plot.png'  # change if desired
)