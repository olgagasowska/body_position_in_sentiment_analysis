
! pip install mediapipe

import os
import csv
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

from google.colab import drive
drive.mount('/content/drive')



input_dir = "/content/drive/MyDrive/MELD-master/data/MELD_Dyadic/train_videos/train_splits"
output_csv = "/content/drive/MyDrive/MELD_train_head_posture_features_4000_8000.csv"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_files = sorted(os.listdir(input_dir))[4000:8000]
print(f"Videos to process: {len(video_files)}")

video_features = []

def calculate_features(landmarks, prev_nose=None):
    """
    Extracts Posture Deviation, Head Tilt Angles (Roll, Pitch, Yaw),
    Raw Positions (x, y, z), and Head Movement Magnitude.
    """
    if not landmarks:
        return None, None, None, None, None

    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Raw Positions
    raw_positions = (nose.x, nose.y, nose.z)

    # Posture Deviation
    posture_deviation = abs(left_shoulder.y - right_shoulder.y)

    # Head Tilt Angles (Roll, Pitch, Yaw) (approximated using keypoints)
    roll = np.arctan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x) * 180 / np.pi
    pitch = np.arctan2(nose.y - (left_shoulder.y + right_shoulder.y) / 2, nose.z) * 180 / np.pi
    yaw = np.arctan2(nose.x - (left_shoulder.x + right_shoulder.x) / 2, nose.z) * 180 / np.pi
    head_tilt_angles = (roll, pitch, yaw)

    # Head Movement Magnitude
    head_movement_magnitude = 0
    if prev_nose:
        head_movement_magnitude = np.sqrt((nose.x - prev_nose[0])**2 + (nose.y - prev_nose[1])**2)

    return raw_positions, posture_deviation, head_tilt_angles, head_movement_magnitude, (nose.x, nose.y)

for video_file in tqdm(video_files, desc="Processing Train Videos"):
    video_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    raw_positions_list, postures, head_tilts, head_movements = [], [], [], []
    prev_nose = None
    frame_count, skip_frames = 0, 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            raw_positions, posture, head_tilt, head_movement, prev_nose = calculate_features(results.pose_landmarks.landmark, prev_nose)
            if raw_positions:
                raw_positions_list.append(raw_positions)
                postures.append(posture)
                head_tilts.append(head_tilt)
                head_movements.append(head_movement)

        frame_count += 1

    cap.release()

    avg_raw_positions = tuple(np.mean(raw_positions_list, axis=0)) if raw_positions_list else (0, 0, 0)
    avg_posture = np.mean(postures) if postures else 0
    avg_head_tilt = tuple(np.mean(head_tilts, axis=0)) if head_tilts else (0, 0, 0)
    avg_head_movement = np.mean(head_movements) if head_movements else 0

    video_features.append({
        "Filename": video_file,
        "Avg Raw Positions (x, y, z)": avg_raw_positions,
        "Avg Posture Deviation": avg_posture,
        "Avg Head Tilt (Roll, Pitch, Yaw)": avg_head_tilt,
        "Avg Head Movement Magnitude": avg_head_movement
    })

print("\nSaving extracted features to CSV...")
with open(output_csv, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Filename", "Avg Raw Positions (x, y, z)", "Avg Posture Deviation", "Avg Head Tilt (Roll, Pitch, Yaw)", "Avg Head Movement Magnitude"])
    writer.writeheader()
    writer.writerows(video_features)

print(f"Features saved to {output_csv}")

text_csv = "/content/drive/MyDrive/MELD-master/data/MELD_Dyadic/train_text_filename_added.csv"
output_csv = "/content/drive/MyDrive/4000_8000_original_data_added.csv"

print("Loading feature and text CSV files...")
features_df = pd.read_csv(output_csv)
text_df = pd.read_csv(text_csv)

print("Merging DataFrames on Filename...")
merged_df = pd.merge(features_df, text_df[["Filename", "Speaker", "Sentiment", "Utterance"]], on="Filename", how="inner")
merged_df.to_csv(output_csv, index=False)
print(f"Merged data saved to {output_csv}")

#removing duplicates
input_file = "/content/drive/MyDrive/MELD_merged_dataset.csv"
output_file = "/content/drive/MyDrive/MELD_deduplicated_dataset.csv"

df = pd.read_csv(input_file)

df_deduplicated = df.drop_duplicates(subset=["Utterance"])

df_deduplicated.to_csv(output_file, index=False)

print(f"Deduplicated dataset saved to {output_file}")



# separating video features
file_path = "/content/drive/MyDrive/MELD_merged_dataset - MELD_merged_dataset.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
s
def extract_tuple(column):
    """Convert string tuples into separate float values."""
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("(") else (0, 0, 0))

df[['Avg_Raw_Pos_X', 'Avg_Raw_Pos_Y', 'Avg_Raw_Pos_Z']] = pd.DataFrame(extract_tuple(df['Avg_Raw_Positions_x_y_z']).tolist(), index=df.index)
df[['Avg_Head_Tilt', 'Avg_Head_Roll', 'Avg_Head_Pitch']] = pd.DataFrame(extract_tuple(df['Avg_Head_Tilt_Roll_Pitch_Yaw']).tolist(), index=df.index)

df = df.drop(columns=['Avg_Raw_Positions_x_y_z', 'Avg_Head_Tilt_Roll_Pitch_Yaw'])

video_feature_cols = ["Avg_Raw_Pos_X", "Avg_Raw_Pos_Y", "Avg_Raw_Pos_Z",
                      "Avg_Posture_Deviation", "Avg_Head_Tilt", "Avg_Head_Roll", "Avg_Head_Pitch",
                      "Avg_Head_Movement_Magnitude"]
df[video_feature_cols] = df[video_feature_cols].astype('float32')

output_file = "/content/drive/MyDrive/MELD_features_cleaned_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")
