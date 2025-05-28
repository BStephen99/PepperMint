import numpy as np
import ast
import glob
import pickle
import pandas as pd
import os

def update_features(data_dict, df):
    # Convert the second column of the DataFrame to string and format as float with 2 decimal places
    df[1] = df[1].astype(float).map('{:.2f}'.format)

    # Iterate through each timestamp in the dictionary
    for timestamp in data_dict.keys():
        # Format the timestamp from the dictionary to have two decimal places
        formatted_timestamp = f"{float(timestamp):.2f}"

        # Find the corresponding rows in the DataFrame
        matching_rows = df[df[1] == formatted_timestamp]

        # Iterate through the entries in the dictionary for this timestamp
        for entry in data_dict[timestamp]:
            person_id_dict = entry['person_id']

            # Find the matching row in the DataFrame for this person_id
            matching_row = matching_rows[matching_rows[2] == person_id_dict]

            if not matching_row.empty:
                # Extract the feature from the DataFrame and convert it to a numpy array of dtype float32
                feature_array = np.array(ast.literal_eval(matching_row.iloc[0, 11]), dtype=np.float32)

                # Update the feature in the dictionary
                entry['feature'] = feature_array

trainSet = "train"

#for g in glob.glob("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AUG/train/*"):
for g in glob.glob("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-OURS/train/*"):
 	video = g.split("/")[-1].split(".")[0]
 	print(video)


 	#if os.path.exists("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AUG4/"+trainSet+"/"+ video +".pkl"):
    #    	continue

 	if os.path.exists("/home2/bstephenson/active-speakers-context/"+trainSet+"_forward2/"+ video +".csv"):
 		with open(g,"rb") as f:
 			data = pickle.load(f)

 		df = pd.read_csv("/home2/bstephenson/active-speakers-context/"+trainSet+"_forward2/"+ video +".csv", header=None)

 		update_features(data, df)
 		with open("/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AUG4/"+trainSet+"/"+video+".pkl","wb") as f:
 			pickle.dump(data, f)
 	else:
 		continue
