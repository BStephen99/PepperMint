import os
import glob
import torch
import pickle  #nosec
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
import ast
import pandas as pd

#misMatchDF = pd.read_csv("/home2/bstephenson/GraVi-T/misMatch/AVAmismatchRows.csv")
#misMatchDF = pd.read_csv("/home2/bstephenson/GraVi-T/misMatch/AVAtrainmismatchRows.csv")

def check_row_exists(df, video_id, timestamp):
    # Filter the DataFrame based on video_id and timestamp
    matching_row = df[(df['video_id'] == video_id) & (df['timestamp'] == timestamp)]

    # Check if any matching row exists
    if not matching_row.empty:
        return True
    else:
        return False


def modify_vector(data, fts):
    # Get the feature vector
    vector = np.array(data[f'{fts}'][0]['feature'])

    # Calculate the length of the vector
    length = len(vector)

    # Create a new vector with the first half of original values and second half as zeros
    modified_vector = np.zeros_like(vector)
    half_length = length // 2
    modified_vector[:half_length] = vector[:half_length]

    return modified_vector

def get_highest_global_id(global_dict):
    max_global_id = -1  # Initialize with a value lower than any expected global_id

    # Iterate through each key and its associated value list in the dictionary
    for value_list in global_dict.values():
        for item in value_list:
            # Check if the 'global_id' exists and update max_global_id if a higher value is found
            if 'global_id' in item:
                max_global_id = max(max_global_id, item['global_id'])

    return max_global_id + 3000000


def clean_global_dict(global_dict):
    # Iterate over each key in the global dictionary
    keys_to_remove = []  # Store keys to remove from the global dictionary
    for key, value_list in global_dict.items():
        #print(value_list)
        # Filter the list to exclude dictionaries with 'pepper' in the person_id
        #filtered_list = [item for item in value_list if "pepper" not in item['person_id']]
        filtered_list = [item for item in value_list if "offscreen" not in item['person_id']]
        #filtered_list = [item for item in filtered_list if [float(c) for c in item['person_box'].split(',')][2] != 0 and [float(c) for c in item['person_box'].split(',')][3] != 0]

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
    with open(data_file, 'rb') as f:
        data = pickle.load(f)  #nosec
    data=clean_global_dict(data)
    #maxGlobal = get_highest_global_id(data)
    #print("maxGlobal",maxGlobal)


    # Get a list of frame_timestamps
    list_fts = sorted(set([float(frame_timestamp) for frame_timestamp in data.keys()]))
    #print(list_fts)

    # Get the time windows where the time span of each window is not greater than "time_span"
    twd_all = _get_time_windows(list_fts, args.time_span)
    #print(twd_all)

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
        for fts in twd:
            #print(fts)
            #print(f'{fts:g}')
            #for entity in data[f'{fts:g}']:
            for entity in data[f'{fts}']:
                #print(fts)
                #print(entity['feature'].shape)
                timestamp.append(fts)
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
                coord.append(np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32))
                label.append(entity['label'])
                person_id.append(entity['person_id'])
                global_id.append(entity['global_id'])
                try:
                    speakerEmb.append(entity['speakerEmb'])
                except:
                    print("no speaker emb")

            """
            if check_row_exists(misMatchDF, video_id, fts):
                timestamp.append(fts)
                pepperSpeaking.append(np.array([0]))
                personSpeaking.append(np.array([1]))
                feature.append(modify_vector(data, fts))
                coord.append(np.array([0, 0, 0, 0], dtype=np.float32))
                label.append(np.array([1]))
                person_id.append(f'{video_id}:offscreen')
                #global_id.append(maxGlobal)
                global_id.append(fts+4000000)
                #maxGlobal += 1
                speakerEmb.append(data[f'{fts}'][0]['speakerEmb'])
            else:
                timestamp.append(fts)
                pepperSpeaking.append(np.array([0]))
                personSpeaking.append(np.array([0]))
                feature.append(modify_vector(data, fts))
                coord.append(np.array([0, 0, 0, 0], dtype=np.float32))
                label.append(np.array([0]))
                person_id.append(f'{video_id}:offscreen')
                #global_id.append(maxGlobal)
                global_id.append(fts+4000000)
                #maxGlobal += 1
                speakerEmb.append(data[f'{fts}'][0]['speakerEmb'])
            """


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

        try:
            graphs = Data(x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                          c = torch.tensor(np.array(coord, dtype=np.float32), dtype=torch.float32),
                          ps = torch.tensor(np.array(pepperSpeaking, dtype=np.float32), dtype=torch.float32),
                          perSpeak = torch.tensor(np.array(pepperSpeaking, dtype=np.float32), dtype=torch.float32),
                          #g = torch.tensor(global_id, dtype=torch.long),
                          g = torch.tensor(global_id, dtype=torch.float32),
                          edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                          edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                          speakerEmb = torch.tensor(np.array(speakerEmb, dtype=np.float32), dtype=torch.float32),
                          y = torch.tensor(np.array(label, dtype=np.float32), dtype=torch.float32))
        except:
            graphs = Data(x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                          c = torch.tensor(np.array(coord, dtype=np.float32), dtype=torch.float32),
                          ps = torch.tensor(np.array(pepperSpeaking, dtype=np.float32), dtype=torch.float32),
                          perSpeak = torch.tensor(np.array(pepperSpeaking, dtype=np.float32), dtype=torch.float32),
                          g = torch.tensor(global_id, dtype=torch.float32),
                          edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                          edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                          #speakerEmb = torch.tensor(np.array(speakerEmb, dtype=np.float32), dtype=torch.float32),
                          y = torch.tensor(np.array(label, dtype=np.float32), dtype=torch.float32))

        num_graph += 1
        print(num_graph)
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
    #for sp in ['AVAtrain', 'AVAval', 'WASDtrain', 'WASDval', 'ours']:
    #for sp in ['train', 'val']:
    #for sp in ['train']:
    for sp in ['val']:
    #for sp in ['ours']:
        path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.ec_mode}_{args.time_span}_{args.tau}/{sp}')
        os.makedirs(path_graphs, exist_ok=True)

        #list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/220926*.pkl')))
        list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{sp}/*.pkl')))
        print(len(list_data_files))

        with Pool(processes=20) as pool:
            num_graph = pool.map(partial(generate_graph, args=args, path_graphs=path_graphs, sp=sp), list_data_files)

        print(f'Graph generation for {sp} is finished (number of graphs: {sum(num_graph)})')
