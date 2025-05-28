import pandas as pd
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_audioclips
from bisect import bisect_left
import subprocess
import ffmpeg
import ast

"""
def draw_bounding_boxes_on_video(df, video_path, output_path):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sort DataFrame by timestamps
    df_sorted = df.sort_values(by='frame_timestamp_groundtruth')

    # Read timestamps from the sorted DataFrame
    timestamps = df_sorted['frame_timestamp_groundtruth'].values
    print(timestamps)

    # Function to find the closest frame index for a given timestamp
    def find_nearest_frame_index(timestamp):
        return bisect_left(timestamps, timestamp)

    # Function to map score to color
    def score_to_color(score):
        return (0, int(255 * score), 0)  # Blue color intensity based on score

    # Function to map label to color
    def label_to_color(label):
        if label == 'not_speaking':
            return (0, 0, 0)  # Blue color for "not_speaking"
        else:
            return (0, 255, 0)  # Orange color for "byplay" or "speaking"

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        nearest_frame_idx = find_nearest_frame_index(timestamp)

        if nearest_frame_idx < len(df_sorted):
            bboxes = df_sorted[df_sorted['frame_timestamp_groundtruth'] == timestamps[nearest_frame_idx]]
            for _, bbox_row in bboxes.iterrows():
                #print(bbox_row)
                if not pd.isna(bbox_row['entity_box_x1_groundtruth']) and not pd.isna(bbox_row['entity_box_y1_groundtruth']) and not pd.isna(bbox_row['entity_box_x2_groundtruth']) and not pd.isna(bbox_row['entity_box_y2_groundtruth']) and not pd.isna(bbox_row['score']):
                    #x1 = int(bbox_row['entity_box_x1_groundtruth'] * width)
                    #y1 = int(bbox_row['entity_box_y1_groundtruth'] * height)
                    #x2 = int(bbox_row['entity_box_x2_groundtruth'] * width)
                    #y2 = int(bbox_row['entity_box_y2_groundtruth'] * height)

                    x1 = int(bbox_row['entity_box_x1_groundtruth'])
                    y1 = int(bbox_row['entity_box_y1_groundtruth'])
                    x2 = int(bbox_row['entity_box_x2_groundtruth'])
                    y2 = int(bbox_row['entity_box_y2_groundtruth'])
                    score = bbox_row['score']
                    color = score_to_color(score)

                    # Draw first bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw second bounding box
                    label = bbox_row['label_groundtruth']
                    label_color = label_to_color(label)
                    x1_second = max(0, int(x1 - (x2 - x1) * 0.1))
                    y1_second = max(0, int(y1 - (y2 - y1) * 0.1))
                    x2_second = min(width, int(x2 + (x2 - x1) * 0.1))
                    y2_second = min(height, int(y2 + (y2 - y1) * 0.1))
                    cv2.rectangle(frame, (x1_second, y1_second), (x2_second, y2_second), label_color, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx >= total_frames:
            break

    cap.release()
    out.release()
"""

def draw_bounding_boxes_on_video(df, video_path, output_path, landmarks, num_speakers):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sort DataFrame by timestamps
    df_sorted = df.sort_values(by='frame_timestamp_groundtruth')

    # Read timestamps from the sorted DataFrame
    timestamps = df_sorted['frame_timestamp_groundtruth'].values
    print(timestamps)

    # Function to find the closest frame index for a given timestamp
    def find_nearest_frame_index(timestamp):
        return bisect_left(timestamps, timestamp)

    # Function to map score to color
    def score_to_color(score):
        return (0, int(255 * score), 0)  # Blue color intensity based on score

    # Function to map label to color
    def label_to_color(label):
        if label == 'not_speaking':
            return (0, 0, 0)  # Blue color for "not_speaking"
        else:
            return (0, 255, 0)  # Orange color for "byplay" or "speaking"

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        print(frame_idx)
        nearest_frame_idx = find_nearest_frame_index(timestamp)

        if nearest_frame_idx < len(df_sorted):
            bboxes = df_sorted[df_sorted['frame_timestamp_groundtruth'] == timestamps[nearest_frame_idx]]
            #print(bboxes["landmarks"])
            for _, bbox_row in bboxes.iterrows():
                # Check for the existence of bounding box and score data
                if not pd.isna(bbox_row['entity_box_x1_groundtruth']) and not pd.isna(bbox_row['entity_box_y1_groundtruth']) and not pd.isna(bbox_row['entity_box_x2_groundtruth']) and not pd.isna(bbox_row['entity_box_y2_groundtruth']) and not pd.isna(bbox_row['score']):
                    x1 = int(bbox_row['entity_box_x1_groundtruth']*width)
                    y1 = int(bbox_row['entity_box_y1_groundtruth']*height)
                    x2 = int(bbox_row['entity_box_x2_groundtruth']*width)
                    y2 = int(bbox_row['entity_box_y2_groundtruth']*height)
                    score = bbox_row['score']
                    color = score_to_color(score)

                    # Draw first bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw second bounding box
                    label = bbox_row['label_groundtruth']
                    label_color = label_to_color(label)
                    x1_second = max(0, int(x1 - (x2 - x1) * 0.1))
                    y1_second = max(0, int(y1 - (y2 - y1) * 0.1))
                    x2_second = min(width, int(x2 + (x2 - x1) * 0.1))
                    y2_second = min(height, int(y2 + (y2 - y1) * 0.1))
                    cv2.rectangle(frame, (x1_second, y1_second), (x2_second, y2_second), label_color, 2)

                    # Draw landmarks
                if landmarks:
                    if not pd.isna(bbox_row['landmarks']):
                        #print(type(bbox_row['landmarks']))
                        if bbox_row['landmarks'] != str(0):
                            landmarks = ast.literal_eval(bbox_row['landmarks'])
                            for i in range(0, len(landmarks), 2):
                                    x_landmark = int(landmarks[i] * width)
                                    y_landmark = int(landmarks[i + 1] * height)
                                    cv2.circle(frame, (x_landmark, y_landmark), 2, color, -1)  # Red color for landmarks

                
                if num_speakers:
                    # Draw text in the upper right-hand corner
                    text = bbox_row['num_speakers']
                    text = f"{text:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 0, 255)  # White color
                    thickness = 2

                    # Get the width and height of the text box
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    # Set the text start position
                    text_x = frame.shape[1] - text_width - 10  # 10 pixels from the right edge
                    text_y = text_height + 10  # 10 pixels from the top edge

                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

        out.write(frame)
        frame_idx += 1
        if frame_idx >= total_frames:
            break

    cap.release()
    out.release()

# Example usage:
#05 no Pepper
#resultsFile = "/home2/bstephenson/GraVi-T/results/results_OURS_processed_with_ALL.csv"
resultsFile = "/home2/bstephenson/GraVi-T/results/results_feature2.csv"
#df = pd.read_csv("/home2/bstephenson/GraVi-T/results/SPELL_ASD_default/df_merged.csv")
#df = pd.read_csv("/home2/bstephenson/GraVi-T/results/df_merged.csv")
df = pd.read_csv(resultsFile)
print(df.shape)
clipNum = "34"
df = df[df["video_id_groundtruth"] == "220927_CLIP_"+clipNum]
print(df.shape)
draw_bounding_boxes_on_video(df, '/home2/bstephenson/220927/220927_CLIP_'+clipNum+'.mp4', '/home2/bstephenson/GraVi-T/results/220927_CLIP_'+clipNum+'_scores.mp4', landmarks=False, num_speakers=False)
