import pickle
import numpy as np

def compare_pickles(input_file, output_file):
    with open(input_file, 'rb') as f:
        original_data = pickle.load(f)

    with open(output_file, 'rb') as f:
        modified_data = pickle.load(f)

    def compare_values(v1, v2, path):
        #print(v1)
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if v1.dtype != v2.dtype or v1.shape != v2.shape:
                print(f"Type or shape mismatch in {path}: original dtype={v1.dtype}, shape={v1.shape}; modified dtype={v2.dtype}, shape={v2.shape}")
        elif isinstance(v1, dict) and isinstance(v2, dict):
            compare_dicts(v1, v2, path)
        elif isinstance(v1, list) and isinstance(v2, list):
            compare_lists(v1, v2, path)
        else:
            if type(v1) != type(v2):
                print(f"Type mismatch in {path}: original type={type(v1)}, modified type={type(v2)}")
            elif v1 != v2:
                print(f"Mismatch in {path}: original={v1}, modified={v2}")

    def compare_dicts(d1, d2, path=""):
        #print("compare_dicts")
        for key in d1.keys():
            if key not in d2:
                print(f"Key {path}.{key} missing in modified data")
                continue

            new_path = f"{path}.{key}" if path else key
            compare_values(d1[key], d2[key], new_path)

        for key in d2.keys():
            if key not in d1:
                print(f"Extra key {path}.{key} in modified data")

    def compare_lists(l1, l2, path):
        #print("compare_lists")
        if len(l1) != len(l2):
            print(f"List length mismatch in {path}: original length={len(l1)}, modified length={len(l2)}")
            return

        for i, (item1, item2) in enumerate(zip(l1, l2)):
            new_path = f"{path}[{i}]"
            compare_values(item1, item2, new_path)

    compare_dicts(original_data, modified_data)

# Example usage
input_file = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AUG/train/Ekwy7wzLfjc.pkl"
output_file = "/home2/bstephenson/GraVi-T/data/features/RESNET18-TSM-AUG2/train/Ekwy7wzLfjc.pkl"
compare_pickles(input_file, output_file)
