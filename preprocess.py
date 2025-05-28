import cv2
import os
from torchvision import transforms
from PIL import Image
import torch
from pydub import AudioSegment
import numpy as np
from models_stage1_tsm import *

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to extract frames
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    return [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder)) if f.endswith('.jpg')]

# Function to preprocess frames
def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        image = Image.open(frame).convert('RGB')
        image = transform(image)
        processed_frames.append(image)
    return torch.stack(processed_frames)

# Function to extract audio
def extract_audio(video_path, output_audio_path):
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(output_audio_path, format="wav")

# Function to preprocess audio
def preprocess_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / np.iinfo(audio.sample_width * 8).max  # Normalize
    return torch.tensor(samples).unsqueeze(0)

# Paths
video_path = '/home2/bstephenson/220929/220929_CLIP_02.mp4'
frames_folder = 'frames'
audio_path = 'audio.wav'
pretrained_weights_path = 'resnet18-tsm-aug.pth'

print("start")
# Extract and preprocess data
frames = extract_frames(video_path, frames_folder)
processed_frames = preprocess_frames(frames)
extract_audio(video_path, audio_path)
processed_audio = preprocess_audio(audio_path)

# Ensure the audio tensor is of type Float
processed_audio = processed_audio.float()  # <-- Ensure it's a Float tensor

# Reshape audio tensor to match the model's expected input shape
# Assuming the model expects the shape [batch_size, channels, height, width]
# Here we assume single channel audio (mono) and treat it as 1D data along the width dimension
processed_audio = processed_audio.unsqueeze(0).unsqueeze(0)

# Debug: Print shape of processed frames before repeat operation
print("Processed frames shape before repeat:", processed_frames.shape)

# Ensure the frames tensor has the correct number of dimensions
rgb_stack_size = 11
if len(processed_frames.shape) == 4:
    # Expand dimensions to add batch size and stack size
    processed_frames = processed_frames.unsqueeze(0)

# Debug: Print shape after unsqueeze
print("Processed frames shape after unsqueeze:", processed_frames.shape)

# Check if the number of frames is sufficient for rgb_stack_size
if processed_frames.shape[1] < rgb_stack_size:
    raise ValueError("Not enough frames to match the required stack size")

# Use the first 11 frames if more than 11 frames are extracted
processed_frames = processed_frames[:, :rgb_stack_size, :, :, :]

# Debug: Print shape after slicing
print("Processed frames shape after slicing:", processed_frames.shape)

# Load the model
model = resnet18_two_streams_forward(pretrained_weights_path=pretrained_weights_path, rgb_stack_size=11, num_classes=2)

# Set the model to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    outputs = model(processed_audio, processed_frames)
    x, aux_a, aux_v, stream_feats = outputs

# The extracted features
print("Extracted features shape:", stream_feats.shape)
