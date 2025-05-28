import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt


width = 1920
height = 1080

def parse_person_box(box_str):
    return torch.tensor([float(x) for x in box_str.split(',')], dtype=torch.float32)

class SpeakerDataset(Dataset):
    def __init__(self, pkl_folder):
        self.data = []
        
        for file in os.listdir(pkl_folder)[:]:
            #print(file)
            if file.endswith('.pkl'):
                with open(os.path.join(pkl_folder, file), 'rb') as f:
                    data_dict = pickle.load(f)
                    
                    for timestamp, people in data_dict.items():
                        if len(people) < 2:
                            continue  # Skip if fewer than two candidates
                        
                        speakers = [p for p in people if p['label'][0] == 1 and p['person_box'] != '0.0,0.0,0.0,0.0']
                        listeners = [p for p in people if p['label'][0] == 0 and p['person_box'] != '0.0,0.0,0.0,0.0']
                        if not speakers or not listeners:
                            continue  # Skip if no speaker or listener
                        
                        # Select the speaker with the largest bounding box
                        speakers.sort(key=lambda p: (float(p['person_box'].split(',')[2]) - float(p['person_box'].split(',')[0])) * 
                                                   (float(p['person_box'].split(',')[3]) - float(p['person_box'].split(',')[1])), 
                                     reverse=True)
                        main_speaker = speakers[0]
                        
                        for listener in listeners:
                            # Store the listener, speaker, filename, and timestamp
                            self.data.append((listener, main_speaker, file, timestamp, listener['person_id']))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        listener, speaker, file, timestamp, person_id = self.data[idx]
        listener_feature = torch.tensor(listener['feature'], dtype=torch.float32)
        listener_box = parse_person_box(listener['person_box'])
        speaker_box = parse_person_box(speaker['person_box'])
        #person_id = listener['person_id']
        
        # Return the features, bounding boxes, filename, and timestamp
        return listener_feature, listener_box, speaker_box, file, timestamp, person_id

class SpeakerPredictor(nn.Module):
    def __init__(self, feature_dim=1024):
        super(SpeakerPredictor, self).__init__()
        self.fc1 = nn.Linear(feature_dim + 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # Output speaker bbox coordinates
        self.relu = nn.ReLU()

    def forward(self, listener_feature, listener_box):
        x = torch.cat((listener_feature, listener_box), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage Example
pkl_folder = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL/WASDtrain"
dataset = SpeakerDataset(pkl_folder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SpeakerPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=6, save_path="speaker_coord_model.pth"):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for listener_feature, listener_box, speaker_box, file, timestamp, person_id in dataloader:
            print(listener_feature[0])
            optimizer.zero_grad()
            predicted_speaker_box = model(listener_feature, listener_box)
            loss = criterion(predicted_speaker_box, speaker_box)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Start training
train_model(model, dataloader, criterion, optimizer)

# Evaluation function
def evaluate_model(model, test_pkl_folder):
    test_dataset = SpeakerDataset(test_pkl_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for listener_feature, listener_box, speaker_box, file, timestamp, person_id in test_dataloader:
            predicted_speaker_box = model(listener_feature, listener_box)
            print("predicted", predicted_speaker_box[1])
            print("real", speaker_box[1])
            loss = criterion(predicted_speaker_box, speaker_box)
            total_loss += loss.item()
    
    print(f"Test Loss: {total_loss / len(test_dataloader):.4f}")

model.load_state_dict(torch.load("speaker_coord_model.pth"))
model.eval()
print("Loaded Model")

# Test set evaluation
test_pkl_folder = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL/WASDval"
print("Evaluating model on test set...")
evaluate_model(model, test_pkl_folder)



# Visualization function
def visualize_predictions(listener_box, gt_speaker_box, pred_speaker_box, canvas_size=(1920, 1080), save_path="visualization.png"):
    """Visualizes listener, ground truth speaker, and predicted speaker bounding boxes on a black canvas and saves the image."""
    img = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)  # Black canvas
    

    def to_pixel_coords(box):
        return int(box[0]*width), int(box[1]*height), int(box[2]*width), int(box[3]*height)
    
    listener_box = to_pixel_coords(listener_box)
    print(listener_box)
    gt_speaker_box = to_pixel_coords(gt_speaker_box)
    print(gt_speaker_box)
    pred_speaker_box = to_pixel_coords(pred_speaker_box)
    print(pred_speaker_box)
    
    # Draw bounding boxes
    cv2.rectangle(img, listener_box[:2], listener_box[2:], (255, 0, 0), 2)  # Blue for listener
    cv2.rectangle(img, gt_speaker_box[:2], gt_speaker_box[2:], (0, 255, 0), 2)  # Green for ground truth speaker
    cv2.rectangle(img, pred_speaker_box[:2], pred_speaker_box[2:], (0, 0, 255), 2)  # Red for predicted speaker
    
    # Save image
    cv2.imwrite(save_path, img)
    print(f"Visualization saved to {save_path}")


import random

def visualize_random_sample(model, test_pkl_folder):
    test_dataset = SpeakerDataset(test_pkl_folder)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty.")
        return

    # Select a random index
    random_idx = random.randint(0, len(test_dataset) - 1)

    # Get the sample
    listener_feature, listener_box, gt_speaker_box, file, timestamp, person_id = test_dataset[random_idx]
    print(file, timestamp)

    # Predict the speaker bbox
    model.eval()
    with torch.no_grad():
        pred_speaker_box = model(listener_feature.unsqueeze(0), listener_box.unsqueeze(0)).squeeze(0)

    # Visualize
    visualize_predictions(listener_box.numpy(), gt_speaker_box.numpy(), pred_speaker_box.numpy())

# Example usage
visualize_random_sample(model, test_pkl_folder)



import os
import cv2
import torch
import pickle
import numpy as np
import random
from torchvision import transforms
from moviepy.editor import ImageSequenceClip
import csv

def draw_bbox(frame, bbox, color, label):
    """Draw a bounding box with a label on a frame."""
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def create_video_visualization(model, test_pkl_folder, canvas_size=(1920, 1080), output_video='output_video_speaker.mp4', fps=10):
    """Creates a video showing ground-truth and predicted speaker bounding boxes and saves predictions to CSV."""
    dataset = SpeakerDataset(test_pkl_folder)
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
    
    # Sort dataset by timestamp
    dataset.data.sort(key=lambda x: (x[2], x[4], x[3]))  # Sort by file, then timestamp
    
    # Select a random file
    unique_files = list(set(x[4] for x in dataset.data))
    if not unique_files:
        print("No valid files found in dataset.")
        return
    selected_file = random.choice(unique_files)
    
    # Filter dataset for the selected file
    dataset.data = [x for x in dataset.data if x[4] == selected_file]
    
    frames = []
    results = []
    model.eval()
    with torch.no_grad():
        for listener_feature, listener_box, gt_speaker_box, file, timestamp, listener_id in dataset:
            # Predict speaker bbox
            pred_speaker_box = model(listener_feature.unsqueeze(0), listener_box.unsqueeze(0)).squeeze(0).numpy()
            gt_speaker_box = gt_speaker_box.numpy()
            listener_box = listener_box.numpy()
            
            # Store results for CSV
            results.append([file, timestamp, listener_id, listener_box.tolist(), gt_speaker_box.tolist(), pred_speaker_box.tolist()])
            
            # Create black canvas
            frame = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            
            # Draw bounding boxes
            draw_bbox(frame, listener_box, (255, 0, 0), f"Listener {listener_id}")
            draw_bbox(frame, gt_speaker_box, (0, 255, 0), "GT Speaker")
            draw_bbox(frame, pred_speaker_box, (0, 0, 255), "Pred Speaker")
            
            frames.append(frame)
    
    if not frames:
        print("No frames processed.")
        return
    
    # Save as video
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    clip.write_videofile(output_video, codec='libx264')
    print(f"Video saved as {output_video}")
    
    # Save results to CSV
    csv_filename = f"{selected_file}_predictions.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "timestamp", "listener_id", "listener_box", "gt_speaker_box", "pred_speaker_box"])
        writer.writerows(results)
    
    print(f"Predictions saved as {csv_filename}")

create_video_visualization(model, test_pkl_folder, output_video='output_video_speaker.mp4', fps=10)