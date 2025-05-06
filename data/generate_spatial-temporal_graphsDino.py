import os
import glob
import torch
import pickle  #nosec
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
import pandas as pd
import ast
import joblib

ipca = joblib.load("/home2/bstephenson/GraVi-T/data/pca_model.pkl")


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


addFeat = pd.read_csv("/home2/bstephenson/GraVi-T/avaAllaugmented.csv")
#addFeat = pd.read_csv("/home2/bstephenson/GraVi-T/avaAllaugmentedGaze.csv")
#addFeat = pd.read_csv("/home2/bstephenson/GraVi-T/train_orig_gender_landmarks_update.csv")


def MaleFemaleOrPepper(value):
    if value == "Female":
        return np.array([0])
    elif value == "Male":
        return np.array([1])
    else:
        return np.array([2])

def processLandmarks(value):
    if value == 0 or value=="0":
        #print(value)
        return [0]*34
    else:
        value = ast.literal_eval(value)
        return value

def processBody(values):
    if len(values) == 0:
        #print(value)
        return [0]*1024
    else:
        #print(values)
        value = list(ast.literal_eval(values[0]))
        return value

def processSpeakerEmb(value):
    new_transformed_data = ipca.transform(value.reshape(1, -1))
    print(new_transformed_data.squeeze(0).shape)
    return new_transformed_data.squeeze(0)


def processGaze(pred_x, pred_y, inout):
    if any(x is None for x in [pred_x, pred_y, inout]):
        #print(value)
        return [0]*3
    else:
        value = [pred_x, pred_y, inout]
        return value

def processNumSpeakers(value):
    if len(value)>0:
        return value[0]
    else:
        #print(value)
        #print(type(value))
        return 0

def clean_global_dict(global_dict):
    # Iterate over each key in the global dictionary
    keys_to_remove = []  # Store keys to remove from the global dictionary
    for key, value_list in global_dict.items():
        # Filter the list to exclude dictionaries with 'pepper' in the person_id
        #filtered_list = [item for item in value_list if "pepper" not in item['person_id']]
        filtered_list = [item for item in value_list if "screen" not in item['person_id']]




        if len(filtered_list) == 0:
            # If the filtered list is empty, mark the key for removal
            keys_to_remove.append(key)
        else:
            # Otherwise, update the list in the global dictionary
            global_dict[key] = filtered_list

    # Remove keys marked for deletion from the global dictionary
    for key in keys_to_remove:
        del global_dict[key]

    return global_dict

def _get_time_windows(list_fts, time_span):
    """
    Get the time windows from the list of frame_timestamps
    Each window is a subset of frame_timestamps where its time span is not greater than "time_span"

    e.g.
    input:
        list_fts:    [902, 903, 904, 905, 910, 911, 912, 913, 914, 917]
        time_span:   3
    output:
        twd_all:     [[902, 903, 904], [905], [910, 911, 912], [913, 914], [917]]
    """

    twd_all = []

    start = end = 0
    while end < len(list_fts):
        while end < len(list_fts) and list_fts[end] < list_fts[start] + time_span:
            end += 1

        twd_all.append(list_fts[start:end])
        start = end

    return twd_all


def generate_graph(data_file, args, path_graphs, sp):
        """
        Generate graphs of a single video
        Time span of each graph is not greater than "time_span"
        """

        video_id = os.path.splitext(os.path.basename(data_file))[0]
        #print(video_id)

        #if os.path.exists(os.path.join(path_graphs, f'{video_id}_0001.pt')) or os.path.exists(os.path.join(path_graphs, f'{video_id}_0002.pt')):
        #    print("pass")
        #    return 1

        #load features file STE
        with open(data_file, 'rb') as f:
            data = pickle.load(f)  #nosec

        #print(data[list(data.keys())[0]][0]["person_id"])
        data=clean_global_dict(data)

        # Get a list of frame_timestamps
        list_fts = sorted(set([float(frame_timestamp) for frame_timestamp in data.keys()]))
        #print(list_fts)

        # Get the time windows where the time span of each window is not greater than "time_span"
        twd_all = _get_time_windows(list_fts, args.time_span)
        print(twd_all)


        # Iterate over every time window
        num_graph = 0
        for twd in twd_all:
                # Skip the training graphs without any temporal edges
                if sp == 'train' and len(twd) == 1:
                    continue

                # Get lists of the timestamps, features, coordinates, labels, person_ids, and global_ids for a given time window
                timestamp, feature, coord, label, person_id, global_id = [], [], [], [], [], []
                pepperSpeaking = []
                personSpeaking = []
                speakerEmb = []
                dinoEmb = []
                gender = []
                landmarks = []
                numPredSpeakers = []
                gaze = []
                bodyEmb =[]

                for fts in twd:
                    #print(fts)
                    #print(f'{fts:g}')
                    #for entity in data[f'{fts:g}']:
                    #print(data[f'{fts}'][0])
                    #data[f'{fts}'].append({'person_box':, 'person_id':f'{video_id}outside{fts}', 'global_id':f'{video_id}outside{fts}', 'feature':, 'speakerEmb':, 'label':np.array([2]})
                    #['person_box', 'person_id', 'global_id', 'feature', 'speakerEmb', 'label']
                    for entity in data[f'{fts}']:

                        print(fts, entity['person_id'])

                        criteria =  (addFeat['frame_timestamp'] == fts) & (addFeat['entity_id'] == entity['person_id'])
                        filtered_df = addFeat[criteria]
                        #filtered_df2 = addFeat2[criteria]

                        if (len(filtered_df['landmarks'].values[0]) == 1 and filtered_df["entity_box_x2"].values[0] == 0 and filtered_df["entity_box_y2"].values[0] == 0) and ("pepper" not in entity['person_id']):
                            #print("zero")
                            #print(filtered_df['landmarks'].values[0])
                            #print(entity['person_id'])
                            #print("**************")
                            continue
                        #print(entity['feature'].shape)
                        timestamp.append(fts)
                        gender.append(MaleFemaleOrPepper(filtered_df['gender'].values[0]))
                        #print(processLandmarks(filtered_df['landmarks'].values[0]))

                        landmarks.append(processLandmarks(filtered_df['landmarks'].values[0]))
                        #gaze.append(processGaze(filtered_df['pred_x'].values[0], filtered_df['pred_y'].values[0], filtered_df['inout'].values[0]))
                        #bodyEmb.append(processBody(filtered_df2["embedding"].values))
                        #consistent_length = len(set([len(sublist) for sublist in landmarks])) == 1
                        #print("Consistent length:", consistent_length)
                        #print(np.array(filtered_df['num_predicted_speakers'].values[0]))
                        #numPredSpeakers.append(np.array(filtered_df['num_predicted_speakers'].values[0]))

                        #dinoEmb.append(dinoDict[fts])#.to(torch.float16))
                        #print(dinoDict[fts].dtype)
                        #print(entity['person_id'])

                        #print(entity['label'] == 1)
                        if "pepper" in entity['person_id'] and entity['label'] == 1:
                            #feature.append(np.append(entity['feature'],1))
                            pepperSpeaking.append(np.array([1]))
                            personSpeaking.append(np.array([1]))
                            #print("pepper speaking")
                        elif entity['label'] == 1 or entity['label'] == 2:
                            pepperSpeaking.append(np.array([0]))
                            personSpeaking.append(np.array([1]))
                        else:
                            #feature.append(np.append(entity['feature'],0))
                            pepperSpeaking.append(np.array([0]))
                            personSpeaking.append(np.array([0]))

                        feature.append(entity['feature'])
                        x1, y1, x2, y2 = [float(c) for c in entity['person_box'].split(',')]
                        #coord.append(np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32))
                        coord.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                        label.append(entity['label'])
                        person_id.append(entity['person_id'])
                        global_id.append(entity['global_id'])
                        #speakerEmb.append(entity['speakerEmb'])
                        speakerEmb.append(processSpeakerEmb(entity['speakerEmb']))

                # Get a list of the edge information: these are for edge_index and edge_attr
                node_source = []
                node_target = []
                edge_attr = []
                for i in range(len(timestamp)):
                    #print("time", i)
                    for j in range(len(timestamp)):
                        # Time difference between the i-th and j-th nodes
                        time_diff = timestamp[i] - timestamp[j]

                        # If the edge connection mode is csi, nodes having the same identity are connected across the frames
                        # If the edge connection mode is cdi, temporally-distant nodes with different identities are also connected
                        if args.ec_mode == 'csi':
                            id_condition = person_id[i] == person_id[j]
                        elif args.ec_mode == 'cdi':
                            id_condition = True

                        # The edge ij connects the i-th node and j-th node
                        # Positive edge_attr indicates that the edge ij is backward (negative: forward)
                        if time_diff == 0 or (abs(time_diff) <= args.tau and id_condition):
                            node_source.append(i)
                            node_target.append(j)
                            edge_attr.append(np.sign(time_diff))

        # x: features
        # c: coordinates of person_box
        # g: global_ids
        # edge_index: information on how the graph nodes are connected
        # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
        # y: labels

        #consistent_length = len(set([len(sublist) for sublist in landmarks])) == 1
        #print("Consistent length:", consistent_length)


                graphs = Data(x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                      c = torch.tensor(np.array(coord, dtype=np.float32), dtype=torch.float32),
                      ps = torch.tensor(np.array(pepperSpeaking, dtype=np.float32), dtype=torch.float32),
                      perSpeak = torch.tensor(np.array(personSpeaking, dtype=np.float32), dtype=torch.float32),
                      g = torch.tensor(global_id, dtype=torch.long),
                      #g = np.array(global_id),
                      gender = torch.tensor(np.array(gender, dtype=np.float32), dtype=torch.long),
                      landmarks = torch.tensor(np.array(landmarks, dtype=np.float32), dtype=torch.float32),
                      #bodyEmb = torch.tensor(np.array(bodyEmb, dtype=np.float32), dtype=torch.float32),
                      #gaze = torch.tensor(np.array(gaze, dtype=np.float32), dtype=torch.float32),
                      #numPredSpeakers = torch.tensor(np.array(numPredSpeakers, dtype=np.float32), dtype=torch.float32),
                      edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                      edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                      speakerEmb = torch.tensor(np.array(speakerEmb, dtype=np.float32), dtype=torch.float32),
                      #dinoEmb = torch.tensor(np.array(dinoEmb, dtype=np.float32), dtype=torch.float32),
                      y = torch.tensor(np.array(label, dtype=np.float32), dtype=torch.float32))


                num_graph += 1
                torch.save(graphs, os.path.join(path_graphs, f'{video_id}_{num_graph:04d}.pt'))

        return num_graph


if __name__ == "__main__":
    """
    Generate spatial-temporal graphs from the extracted features
    """

    parser = argparse.ArgumentParser()
    # Default paths for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--features',      type=str,   help='Name of the features', required=True)

    # Two options for the edge connection mode:
    # csi: Connect the nodes only with the same identities across the frames
    # cdi: Connect different identities across the frames
    parser.add_argument('--ec_mode',       type=str,   help='Edge connection mode (csi | cdi)', required=True)
    parser.add_argument('--time_span',     type=float, help='Maximum time span for each graph in seconds', required=True)
    parser.add_argument('--tau',           type=float, help='Maximum time difference between neighboring nodes in seconds', required=True)

    args = parser.parse_args()

    # Iterate over train/val splits
    #addFeat2 = pd.read_csv("/home2/bstephenson/gestsync/220927embeddingsFixed.csv")
    #print(addFeat2.iloc[2])
    #print(list(ast.literal_eval(addFeat2.iloc[2]["embedding"])))

    print ('This process might take a few minutes')
    #for sp in ['val']:
    #for sp in ['ours']:
    for sp in ['train', 'test']:
    #for sp in ["WASDval"]
        path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.ec_mode}_{args.time_span}_{args.tau}/{sp}')
        os.makedirs(path_graphs, exist_ok=True)

        #list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/220929*.pkl')))[:]
        list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/*.pkl')))[:]
        print(len(list_data_files))

        with Pool(processes=20) as pool:
            num_graph = pool.map(partial(generate_graph, args=args, path_graphs=path_graphs, sp=sp), list_data_files)

        print(f'Graph generation for {sp} is finished (number of graphs: {sum(num_graph)})')
