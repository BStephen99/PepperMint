import subprocess
import os
import time
import shutil


#expName = "withWASD"
#expName = "ALL2"
#expName = "byplayGAZE"
#expName= "byplayLAND"
#expName = "byplay2Feats"
#expName = "byplayAudioOnly"
#expName = "byplayVisualOnly"
#expName = "byplayGaze2views"
#expName = "speakEmb"
expName = "forward"


"""
def run_training_multiple_times():
    command = [
        "python3", 
        "tools/train_context_reasoning.py", 
        "--cfg", 
        "configs/active-speaker-detection/ava_active-speaker/SPELL_default.yaml"
    ]
    ckpt_path = "/home2/bstephenson/GraVi-T/results/"+expName+"/ckpt_best.pt"
    results_dir = "/home2/bstephenson/GraVi-T/results/"+expName

    for i in range(1, 6):
        print(i)
        print(f"Running training iteration {i}...")
        subprocess.run(command, check=True)

        # Wait to ensure file is written (optional: adjust timing or polling logic)
        time.sleep(2)

        # Check if file exists and rename it
        if os.path.exists(ckpt_path):
            new_ckpt_path = os.path.join(results_dir, f"ckpt_best{i}.pt")
            shutil.move(ckpt_path, new_ckpt_path)
            print(f"Renamed checkpoint to: {new_ckpt_path}")
        else:
            print(f"Warning: {ckpt_path} not found after run {i}!")



#run_training_multiple_times()


def evaluate_multiple_checkpoints():
    base_cmd = "python3 tools/evaluateOneViews.py --exp_name "+expName+" --eval_type AVA_ASD --modelNum"
    results_dir = "/home2/bstephenson/GraVi-T/results"
    output_file = os.path.join(results_dir, "results_feature.csv")

    for i in range(1, 6):
        checkpoint = f"ckpt_best{i}.pt"
        print(f"Running evaluation for {checkpoint}...")

        # Run the command
        subprocess.run(f"{base_cmd} {checkpoint}", shell=True, check=True)

        # Rename the results file
        new_filename = os.path.join(results_dir, expName, f"{i}.csv")
        if os.path.exists(output_file):
            shutil.move(output_file, new_filename)
            print(f"Renamed results to {new_filename}")
        else:
            print(f"Warning: {output_file} not found after running {checkpoint}")


evaluate_multiple_checkpoints()
"""

def run_training_multiple_times():
    command = [
        "python3", 
        "tools/train_context_reasoning.py", 
        #"tools/train_context_reasoningMulticlass.py",
        "--cfg", 
        #"configs/active-speaker-detection/ava_active-speaker/SPELL_defaultByplay.yaml"
        #"configs/active-speaker-detection/ava_active-speaker/SPELL_SpeakEmb.yaml"
        "configs/active-speaker-detection/ava_active-speaker/SPELL_forward.yaml"
    ]
    ckpt_path = "/home2/bstephenson/GraVi-T/results/"+expName+"/ckpt_best.pt"
    results_dir = "/home2/bstephenson/GraVi-T/results/"+expName

    for i in range(1, 6):
        print(i)
        print(f"Running training iteration {i}...")
        subprocess.run(command, check=True)

        # Wait to ensure file is written (optional: adjust timing or polling logic)
        time.sleep(2)

        # Check if file exists and rename it
        if os.path.exists(ckpt_path):
            new_ckpt_path = os.path.join(results_dir, f"ckpt_best{i}.pt")
            shutil.move(ckpt_path, new_ckpt_path)
            print(f"Renamed checkpoint to: {new_ckpt_path}")
        else:
            print(f"Warning: {ckpt_path} not found after run {i}!")



run_training_multiple_times()



def evaluate_multiple_checkpoints():
    #base_cmd = "python3 tools/evaluateOneViewsByplay.py --exp_name "+expName+" --eval_type AVA_ASD --modelNum"
    base_cmd = "python3 tools/evaluate.py --exp_name "+expName+" --eval_type AVA_ASD --mode pepper --modelNum"
    results_dir = "/home2/bstephenson/GraVi-T/results"
    output_file = os.path.join(results_dir, "results_feature.csv")

    for i in range(1, 6):
        checkpoint = f"ckpt_best{i}.pt"
        print(f"Running evaluation for {checkpoint}...")

        # Run the command
        subprocess.run(f"{base_cmd} {checkpoint}", shell=True, check=True)

        # Rename the results file
        new_filename = os.path.join(results_dir, expName, f"{i}.csv")
        if os.path.exists(output_file):
            shutil.move(output_file, new_filename)
            print(f"Renamed results to {new_filename}")
        else:
            print(f"Warning: {output_file} not found after running {checkpoint}")


evaluate_multiple_checkpoints()