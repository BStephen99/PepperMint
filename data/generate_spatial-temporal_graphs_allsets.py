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

#ipca = joblib.load("/home2/bstephenson/GraVi-T/data/pca_model.pkl")


import warnings
warnings.filterwarnings("ignore", category=UserWarning)




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
    return new_transformed_data.squeeze(0)




def processGaze(gaze):
    if any(x is None for x in gaze):
        #print(value)
        return [0]*3
    else:
        value = list(gaze)
        return value

def processNumSpeakers(value):
    if len(value)>0:
        return value[0]
    else:
        return 0

def clean_global_dict(global_dict):
    # Iterate over each key in the global dictionary
    keys_to_remove = []  # Store keys to remove from the global dictionary
    for key, value_list in global_dict.items():
        # Filter the list to exclude dictionaries with 'pepper' in the person_id
        #filtered_list = [item for item in value_list if "pepper" not in item['person_id']]


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

    

        #load features file STE
        with open(data_file, 'rb') as f:
            data = pickle.load(f)  #nosec


        # Get a list of frame_timestamps
        list_fts = sorted(set([float(frame_timestamp) for frame_timestamp in data.keys()]))
        #print(list_fts)

        # Get the time windows where the time span of each window is not greater than "time_span"
        twd_all = _get_time_windows(list_fts, args.time_span)


        # Iterate over every time window
        num_graph = 0
        for twd in twd_all:
                # Skip the training graphs without any temporal edges
                if sp == 'train' and len(twd) == 1:
                    continue

                # Initialize lists to collect data for a given time window
                timestamp       = []
                feature         = []
                coord           = []
                label           = []
                person_id       = []
                global_id       = []
                pepperSpeaking  = []
                personSpeaking  = []
                speakerEmb      = []
                gender          = []
                landmarks       = []
                numPredSpeakers = []

                for fts in twd:
                    for entity in data[f'{fts}']:

                        print(fts, entity['person_id'])

        
                        timestamp.append(fts)
                    
                        if "pepper" in entity['person_id'] and entity['label'] == 1:
                            pepperSpeaking.append(np.array([1]))
                            personSpeaking.append(np.array([1]))
                        elif entity['label'] == 1 or entity['label'] == 2:
                            pepperSpeaking.append(np.array([0]))
                            personSpeaking.append(np.array([1]))
                        else:
                            pepperSpeaking.append(np.array([0]))
                            personSpeaking.append(np.array([0]))

                        feature.append(entity['feature'])
                        #featureHigh.append(entity['featureHigh'])
                        x1, y1, x2, y2 = [float(c) for c in entity['person_box'].split(',')]
                        coord.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                 
                        label.append(entity['label'])

                        person_id.append(entity['person_id'])
                        global_id.append(entity['global_id'])
                        #if 'gender' in entity:
                        #print(entity['gender'])
                        gender.append(MaleFemaleOrPepper(entity['gender']))
                        
                        
                        landmarks.append(processLandmarks(entity['landmarks']))
                        #speakerEmb.append(processSpeakerEmb(entity['speakerEmb']))
                        speakerEmb.append(entity['speakerEmb'])
        

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
                      gender = torch.tensor(np.array(gender, dtype=np.float32), dtype=torch.long), #not in all
                      landmarks = torch.tensor(np.array(landmarks_back, dtype=np.float32), dtype=torch.float32), #not in all
                      gaze = torch.tensor(np.array(gaze, dtype=np.float32), dtype=torch.float32), #not in all
                      #numPredSpeakers = torch.tensor(np.array(numPredSpeakers, dtype=np.float32), dtype=torch.float32),
                      edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                      edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                      speakerEmb = torch.tensor(np.array(speakerEmb, dtype=np.float32), dtype=torch.float32),
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
 

    print ('This process might take a few minutes')
    #for sp in ['val']:
    #for sp in ['ours']:
    #for sp in ["AVAtrain", "WASDtrain", 'train', 'test']:
    #for sp in ["WASDtrain", 'train', 'test']:
    #for sp in ['train', 'test']:
    #for sp in ['test']:
    #for sp in ["AVAtrain", "WASDtrain"]:
    for sp in ["WASDtrainLaugh"]:
        path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.ec_mode}_{args.time_span}_{args.tau}/{sp}')
        os.makedirs(path_graphs, exist_ok=True)

        #list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/220929*.pkl')))[:]
        list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/*.pkl')))[:]
        print(len(list_data_files))

        with Pool(processes=20) as pool:
            num_graph = pool.map(partial(generate_graph, args=args, path_graphs=path_graphs, sp=sp), list_data_files)

        print(f'Graph generation for {sp} is finished (number of graphs: {sum(num_graph)})')
