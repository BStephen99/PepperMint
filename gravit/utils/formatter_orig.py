import os
import glob
import torch
import pickle  #nosec
import pandas as pd
import numpy as np

#addFeat = pd.read_csv("/home2/bstephenson/GraVi-T/avaAllaugmented.csv")
#addFeat = pd.read_csv("/home2/bstephenson/GraVi-T/avaAllaugmentedGaze.csv")

def check_row_exists(df, video_id, timestamp):
    # Filter the DataFrame based on video_id and timestamp
    matching_row = df[(df['video_id'] == video_id) & (df['timestamp'] == timestamp)]

    # Check if any matching row exists
    if not matching_row.empty:
        return True
    else:
        return False

def get_highest_global_id(global_dict):
    max_global_id = -1  # Initialize with a value lower than any expected global_id

    # Iterate through each key and its associated value list in the dictionary
    for value_list in global_dict.values():
        for item in value_list:
            # Check if the 'global_id' exists and update max_global_id if a higher value is found
            if 'global_id' in item:
                max_global_id = max(max_global_id, item['global_id'])

    return max_global_id


def get_formatting_data_dict(cfg):
    """
    Get a dictionary that is used to format the results following the formatting rules of the evaluation tool
    """

    root_data = cfg['root_data']
    dataset = cfg['dataset']
    data_dict = {}

    if 'AVA' in cfg['eval_type']:
        # Get a list of the feature files
        features = '_'.join(cfg['graph_name'].split('_')[:-3])
        print("features", features)
        #print(os.path.join(root_data, f'features/{features}/WASDval/*'))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/test/*.pkl')))
        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/WASDval/*')))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/val_AVA/*.pkl')))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/val/*.pkl')))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/ours/*.pkl')))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/ours/220927*.pkl')))
        #list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/ours/220928*.pkl'))) + sorted(glob.glob(os.path.join(root_data, f'features/{features}/ours/220929*.pkl'))) + sorted(glob.glob(os.path.join(root_data, f'features/{features}/ours/220926*.pkl')))

        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]

            with open(data_file, 'rb') as f:
                data = pickle.load(f) #nosec

            #maxGlobal = get_highest_global_id(data)
            #print("maxGlobal",maxGlobal)

            # Get a list of frame_timestamps
            list_fts = sorted([float(frame_timestamp) for frame_timestamp in data.keys()])

            # Iterate over all the frame_timestamps and retrieve the required data for evaluation
            for fts in list_fts:
                #frame_timestamp = f'{fts:g}'
                frame_timestamp = f'{fts}'
                for entity in data[frame_timestamp]:
                    #if '0BRxm7G1acw_1419-1449' in entity["person_id"]:
                    #    print(fts)
                    #entity['landmarks'] = '0'
                    #print(entity['person_id'])
                    #print(entity['person_box'])
                    data_dict[entity['global_id']] = {'video_id': video_id,
                                                      'frame_timestamp': frame_timestamp,
                                                      'person_box': entity['person_box'],
                                                      'person_id': entity['person_id'],
                                                      'label': entity['label']}
                                                      #'landmarks': entity['landmarks']}
                #if check_row_exists(misMatchDF, video_id, fts):

                """
                data_dict[fts+4000000] = {'video_id': video_id,
                                        'frame_timestamp': frame_timestamp,
                                        'person_box': "0.0,0.0,0.0,0.0",
                                        'person_id': f'{video_id}:offscreen'}

                maxGlobal += 1

                else:
                    data_dict[maxGlobal] = {'video_id': video_id,
                                                      'frame_timestamp': frame_timestamp,
                                                      'person_box': np.array([0, 0, 0, 0], dtype=np.float32),
                                                      'person_id': f'{video_id}:offscreen'}
                    maxGlobal += 1
                """
    elif 'AS' in cfg['eval_type']:
        # Build a mapping from action ids to action classes
        data_dict['actions'] = {}
        with open(os.path.join(root_data, 'annotations', dataset, 'mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                data_dict['actions'][int(aid)] = cls

        # Get a list of all video ids
        data_dict['all_ids'] = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/groundTruth'))])

    return data_dict


def get_formatted_preds(cfg, logits, g, data_dict):
    """
    Get a list of formatted predictions from the model output, which is used to compute the evaluation score
    """

    eval_type = cfg['eval_type']
    preds = []
    if 'AVA' in eval_type:
        # Compute scores from the logits
        scores_all = torch.sigmoid(logits.detach().cpu()).numpy()
        #scores_all = torch.sigmoid(logits[:,1].detach().cpu()).numpy()

        # Iterate over all the nodes and get the formatted predictions for evaluation
        for scores, global_id in zip(scores_all, g):
            #if global_id in data_dict:
                if global_id in data_dict:
                    data = data_dict[global_id]
                    #print(data["label"])
                else:
                    continue
                #print(data)
                #if data["label"] == [0]:
                #    continue
                if "pepper" in data['person_id']:
                    continue

               

                #if data["landmarks"] == '0' and data["person_box"]=='0,0,0,0':
                    #print(data['person_id'])
                    #print("no landmarks")
                    #continue
                #if data['person_box']=='0,0,0,0':
                #   continue
                video_id = data['video_id']
                frame_timestamp = float(data['frame_timestamp'])
                x1, y1, x2, y2 = [float(c) for c in data['person_box'].split(',')]

                #criteria =  (addFeat['frame_timestamp'] == frame_timestamp) & (addFeat['entity_id'] == data['person_id'])
                #filtered_df = addFeat[criteria]
                #landmarks = filtered_df['landmarks'].values[0]
                #if landmarks == '0':
                #    print("zero")
                #    continue

                if eval_type == 'AVA_ASD':
                    # Line formatted following Challenge #2: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                    person_id = data['person_id']
                    score = scores.item()
                    pred = [video_id, frame_timestamp, x1, y1, x2, y2, 'SPEAKING_AUDIBLE', person_id, score]
                    preds.append(pred)

                elif eval_type == 'AVA_AL':
                    # Line formatted following Challenge #1: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                    for action_id, score in enumerate(scores, 1):
                        pred = [video_id, frame_timestamp, x1, y1, x2, y2, action_id, score]
                        preds.append(pred)
    elif 'AS' in eval_type:
        tmp = logits
        if cfg['use_ref']:
            tmp = logits[-1]

        tmp = torch.softmax(tmp.detach().cpu(), dim=1).max(dim=1)[1].tolist()

        # Upsample the predictions to fairly compare with the ground-truth labels
        preds = []
        for pred in tmp:
            preds.extend([data_dict['actions'][pred]] * cfg['sample_rate'])

        # Pair the final predictions with the video_id
        (g,) = g
        video_id = data_dict['all_ids'][g]
        preds = [(video_id, preds)]

    elif 'VS' in eval_type:
        tmp = logits
        tmp = torch.sigmoid(tmp.squeeze().cpu()).numpy().tolist()
        (g,) = g
        preds.append([f"video_{g}", tmp])

    return preds
