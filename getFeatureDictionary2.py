import pandas as pd
import numpy as np
import pickle
import ast
import glob
import os

def create_and_save_dict(df1, df2, df3, filename, global_count):
    result_dict = {}

    #global_count = 0
    width = 1920
    height = 1080

    for _, row in df1.iterrows():
        #print(row)
        key = str(row['frame_timestamp'])
        if row['entity_box_x2_back'] > 0:
            #person_box = f"{row['entity_box_x1']/width},{row['entity_box_y1']/height},{row['entity_box_x2']/width},{row['entity_box_y2']/height}"
            person_box = f"{row['entity_box_x1_back']},{row['entity_box_y1_back']},{row['entity_box_x2_back']},{row['entity_box_y2_back']}"
        else:
            #print("None")
            #print(row)
            #continue
            person_box = f"{0},{0},{0},{0}"
        if row['entity_box_x2_high'] > 0:
            person_box2 = f"{row['entity_box_x1_high']},{row['entity_box_y1_high']},{row['entity_box_x2_high']},{row['entity_box_y2_high']}"
        else:
            #print("None")
            #print(row)
            #continue
            person_box2 = f"{0},{0},{0},{0}"

        person_id = row['entity_id']
        gender = row['gender']
        #gaze = row['gaze_pred_x']/width, row['gaze_pred_y']/height, row['gaze_inout']
        gaze = row['gaze_pred_x'], row['gaze_pred_y'], row['gaze_inout']
        #print(gaze)
        landmarks = row['landmarks_back']
        landmarks2 = row['landmarks_high']
        #label = np.array([row['label_id_back']])
        label = np.array([row['byplay']])
        global_id = global_count
        global_count += 1

        # Find the corresponding feature in df2
        feature_row = df2[(df2[df2.columns[0]] == row['video_id']) &
                          (df2[df2.columns[1]] == row['frame_timestamp']) &
                          (df2[df2.columns[2]] == row['entity_id'])]

        if not feature_row.empty:
            feature_str = feature_row.iloc[0][df2.columns[10]] #get feature
            feature_list = ast.literal_eval(feature_str)  # Convert string to list of floats
            feature = np.array(feature_list, dtype=np.float32)
            #print(feature.shape)
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
            feature_list2 = ast.literal_eval(feature_str2)  # Convert string to list of floats
            feature2 = np.array(feature_list2, dtype=np.float32)
            #print(feature.shape)
            #speakerEmb2 = feature_row2.iloc[0][df3.columns[11]]
            #speakerEmb2 = ast.literal_eval(speakerEmb2)
            #speakerEmb2 = np.array(speakerEmb2, dtype=np.float32)
        else:
            feature2 = np.zeros(1024) #np.array([])
            #speakerEmb2 = np.zeros(192)

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
            'label': label
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

df1 = pd.read_csv('/home2/bstephenson/GraVi-T/annotations.csv', dtype={26: str})
print(df1.shape)
df1 = df1[df1["video_id"]!="220928_CLIP_13A"]
print(df1.shape)
df1 = df1[df1["set"]=="test"]
df1["landmarks_high"] = df1["landmarks_high"].fillna("0")
df1["landmarks_back"] = df1["landmarks_back"].fillna("0")

for v in df1["video_id"].unique()[:]:
    #if v != "220926_CLIP_50":
    #    continue
    vdf1 = df1[df1["video_id"]==v]
    #df2 = pd.read_csv('/home2/bstephenson/active-speakers-context/oursBackTest/'+v+'.csv', header=None)
    #df3 = pd.read_csv('/home2/bstephenson/active-speakers-context/oursHighTest/'+v+'.csv', header=None)
    #df2 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/oursBackTrain/'+v+'.csv')
    #df3 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/oursHighTrain/'+v+'.csv')
    #df2 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/AVAoursBackTest/'+v+'.csv')
    #df3 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/AVAoursHighTest/'+v+'.csv')
    df2 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/WASDoursBackTest/'+v+'.csv')
    df3 = pd.read_csv('/home2/bstephenson/active-speakers-context/justOurs/WASDoursHighTest/'+v+'.csv')

    #filename = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-OURS/train/"+v+".pkl"
    #filename = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AVA/test/"+v+".pkl"
    #filename = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/test/"+v+".pkl"
    filename = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-WASD/test/"+v+".pkl"
    #filename = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-ALL2/WASDTrain/"+v+".pkl"

    global_count = create_and_save_dict(vdf1, df2, df3, filename, global_count)