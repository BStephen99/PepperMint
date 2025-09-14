import pandas as pd
import numpy as np
import pickle
import ast
import glob
import os
import time


def parse_feature_str(feature_str):
    cleaned = feature_str.replace("np.float32", "")
    parsed = ast.literal_eval(cleaned)
    # flatten in case it's nested
    return [float(x) for sub in (parsed if isinstance(parsed[0], (list, tuple)) else [parsed]) for x in (sub if isinstance(sub, (list, tuple)) else [sub])]


def create_and_save_dict(df1, df2, filename, global_count):
    result_dict = {}

    #global_count = 0
    width = 1920
    height = 1080

    for _, row in df1.iterrows():
        #print(row)
        key = str(row['frame_timestamp'])
        if row['entity_box_x2'] > 0 or row['entity_box_x2'] > 0:
            person_box = f"{row['entity_box_x1']},{row['entity_box_y1']},{row['entity_box_x2']},{row['entity_box_y2']}"
        else:
            print("None")
            print(row)
            #continue
            person_box = f"{0},{0},{0},{0}"
        person_id = row['entity_id']
        label = np.array([row['label_id']])
        global_id = global_count
        global_count += 1



        feature_row = df2[(df2[df2.columns[0]] == row['video_id']) &
                          (df2[df2.columns[1]] == row['frame_timestamp']) &
                          #(np.isclose(df2[df2.columns[1]], row['frame_timestamp'], atol=1e-2)) &
                          (df2[df2.columns[2]] == row['entity_id'])]
        
   

        if not feature_row.empty:
            feature_str = feature_row.iloc[0][df2.columns[10]] #get feature
            feature_list = parse_feature_str(feature_str)

            feature = np.array(feature_list, dtype=np.float32)
            speakerEmb = feature_row.iloc[0][df2.columns[11]]
            speakerEmb = ast.literal_eval(speakerEmb)
            speakerEmb = np.array(speakerEmb, dtype=np.float32)
        else:
            print(row['video_id'], row['frame_timestamp'], row['entity_id'], 'notfound')
            feature = np.zeros(1024) #np.array([])
            speakerEmb = np.zeros(192)


        entry = {
            'person_box': person_box,
            'person_id': person_id,
            'global_id': global_id,
            'feature': feature,
            'speakerEmb':speakerEmb,
            'label': label
        }

        if key not in result_dict:
            result_dict[key] = []

        if person_id not in [ent["person_id"] for ent in result_dict[key]]:
            result_dict[key].append(entry)
        else:
            print("duplicate******************************")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(result_dict, f)

    return global_count





global_count = 0



CSV_FILE = "path_to_ava_csv.csv"

FEAT_DIR = "path_to_ava_features/allAVAval"


FEATURES_BASE = "./data/features/RESNET18-TSM-ALL"
FEATURES_SUBDIR = "AVAval"


# ==========================
# Load data
# ==========================
df1 = pd.read_csv(CSV_FILE)
allVideos = df1["video_id"].unique()

# ==========================
# Loop over clips
# ==========================
for g in glob.glob(os.path.join(FEAT_DIR, "*.csv")):
    clip = os.path.basename(g).replace(".csv", "")
    print(clip)

    filename = os.path.join(FEATURES_BASE, FEATURES_SUBDIR, f"{clip}.pkl")


    df1a = df1[df1["video_id"] == clip]

    df2_path = os.path.join(FEAT_DIR, f"{clip}.csv")
    if not os.path.exists(df2_path):
        print(f"[WARNING] Missing file: {df2_path} — skipping {clip}")
        continue

    df2 = pd.read_csv(df2_path, header=None)

    global_count = create_and_save_dict(df1a, df2, filename, global_count)
    
