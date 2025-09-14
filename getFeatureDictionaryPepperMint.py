import pandas as pd
import numpy as np
import pickle
import ast
import glob
import os


def parse_feature_str(feature_str):
    cleaned = feature_str.replace("np.float32", "")
    parsed = ast.literal_eval(cleaned)
    # flatten in case it's nested
    return [float(x) for sub in (parsed if isinstance(parsed[0], (list, tuple)) else [parsed]) for x in (sub if isinstance(sub, (list, tuple)) else [sub])]

def create_and_save_dict(df1, df2, df3, filename, global_count):
    result_dict = {}

    #global_count = 0
    width = 1920
    height = 1080

    for _, row in df1.iterrows():
        #print(row)
        key = str(row['frame_timestamp'])
        if row['entity_box_x2_back'] > 0:
            person_box = f"{row['entity_box_x1_back']},{row['entity_box_y1_back']},{row['entity_box_x2_back']},{row['entity_box_y2_back']}"
        else:
            person_box = f"{0},{0},{0},{0}"
        if row['entity_box_x2_high'] > 0:
            person_box2 = f"{row['entity_box_x1_high']},{row['entity_box_y1_high']},{row['entity_box_x2_high']},{row['entity_box_y2_high']}"
        else:
            person_box2 = f"{0},{0},{0},{0}"

        person_id = row['entity_id']
        gender = row['gender']
        gaze = row['gaze_pred_x'], row['gaze_pred_y'], row['gaze_inout']
        landmarks = row['landmarks_back']
        landmarks2 = row['landmarks_high']
        label = np.array([row['label_id']])
        laugh = np.array([row['laugh_speaker']])
        global_id = global_count
        global_count += 1

        # Find the corresponding feature in df2
        feature_row = df2[(df2[df2.columns[0]] == row['video_id']) &
                          (df2[df2.columns[1]] == row['frame_timestamp']) &
                          (df2[df2.columns[2]] == row['entity_id'])]

        if not feature_row.empty:
            #feature_list = feature_row.iloc[0][df2.columns[10]] #get feature
            feature_str = feature_row.iloc[0][df2.columns[10]] #get feature
            feature_list = parse_feature_str(feature_str)
            feature = np.array(feature_list, dtype=np.float32)
            speakerEmb = feature_row.iloc[0][df2.columns[11]]
            speakerEmb = ast.literal_eval(speakerEmb)
            speakerEmb = np.array(speakerEmb, dtype=np.float32)
        else:
            feature = np.zeros(1024) #np.array([])
            speakerEmb = np.zeros(192)


        feature_row2 = df3[(df3[df3.columns[0]] == row['video_id']) &
                          (df3[df3.columns[1]] == row['frame_timestamp']) &
                          (df3[df3.columns[2]] == row['entity_id'])]


        if not feature_row2.empty:
            feature_str2 = feature_row2.iloc[0][df3.columns[10]]
            feature_list2 = parse_feature_str(feature_str2)
            feature2 = np.array(feature_list2, dtype=np.float32)
        else:
            feature2 = np.zeros(1024)
        

        entry = {
            'person_box': person_box,
            'person_boxHigh': person_box2,
            'person_id': person_id,
            'global_id': global_id,
            'feature': feature,
            'featureHigh': feature2,
            'speakerEmb':speakerEmb,
            'gender': gender,
            'landmarks_back': landmarks,
            'landmarks_high': landmarks2,
            'gaze': gaze,
            'label': label,
            'laugh': laugh
        }

        if key not in result_dict:
            result_dict[key] = []

        if person_id not in [ent["person_id"] for ent in result_dict[key]]:
            result_dict[key].append(entry)
        else:
            print("duplicate******************************")

    with open(filename, 'wb') as f:
        print(filename)
        pickle.dump(result_dict, f)

    return global_count


global_count = 0



mode = "train"  # "train" or "test"
CSV_PATH = #'path_to_csv/annotations.csv'
BASE_DIR = path_to_feature_dir
BACK_DIR = os.path.join(BASE_DIR, "allOursBack")
HIGH_DIR = os.path.join(BASE_DIR, "allOursHigh")

FEATURES_BASE = "./data/features/RESNET18-TSM-ALL"  #path to feature dictionary directory


df1 = pd.read_csv(CSV_PATH)
df1 = df1[df1["set"] == mode]

# ==========================
# Loop over videos
# ==========================
for v in df1["video_id"].unique():
    vdf1 = df1[df1["video_id"] == v]

    df2_path = os.path.join(BACK_DIR, f"{v}.csv")
    df2 = pd.read_csv(df2_path, header=None)

    df3_path = os.path.join(HIGH_DIR, f"{v}.csv")
    if os.path.exists(df3_path):
        df3 = pd.read_csv(df3_path, header=None)
    else:
        print(f"File not found: {df3_path} — using default values.")
        # Create empty df3 with expected number of columns (at least 11, since columns[10] is accessed)
        df3 = pd.DataFrame(columns=list(range(12)))  # Adjust number of columns if needed

    filename = os.path.join(FEATURES_BASE, mode, f"{v}.pkl")

    global_count = create_and_save_dict(vdf1, df2, df3, filename, global_count)














