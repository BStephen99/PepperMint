
## Credit

This repository is largely based on the original **SPELL** model and codebase for active speaker detection developed by Min et al., available at [https://github.com/IntelLabs/GraVi-T](https://github.com/IntelLabs/GraVi-T).

Minor modifications have been made to adapt the code for specific use cases.

If you use this code or build upon it, please consider citing the original paper:

@inproceedings{min2022learning,
title={Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection},
author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
booktitle={European Conference on Computer Vision},
pages={371--387},
year={2022},
organization={Springer}
}







exp_name: byplayGaze2views
model_name: SPELLBYPLAYGAZE   
graph_name: RESNET18-TSM-ALL2_csi_30.0_0.9
loss_name: ce  , bce_logit
use_spf: True
use_ref: False
num_modality: 2
channel1: 64
channel2: 16

proj_dim: 64
final_dim: 4
num_att_heads: 0
dropout: 0.3
lr: 0.0005
wd: 0
batch_size: 16
sch_param: 10
num_epoch: 70

training_sets: ["train"]
test_sets: ["test"]

gender: False
gaze: True
twoView: True
numPredSpeakers: False
multiclass: True
classIndex: 1  # multiclass model

csv_path: "/home2/bstephenson/GraVi-T/annotations.csv"

positiveLabels:  ["speaking_to_pepper", "speaking_to_human"] #labels for the positive classes in the csv
genderClass: False  #use gender predictor model

laugh_not_speech: False  #if True, do not count laughter as speech



#mode = pepper, wasd, byplay

 SPELLBYPLAY, SPELLBYPLAYGAZE, SPELLBYPLAYLAND, SPELLBYPLAY2FEATS, SPELLBYPLAYAUDIOONLY, SPELLVISONLYBYPLAY


#to train addressee detection model
python3 tools/train_context_reasoning_multiclass.py --cfg cfg_path


python3 tools/evaluate.py --exp_name Test --eval_type AVA_ASD --mode pepper --modelNum None


python3 data/generate_spatial-temporal_graphs_peppermint.py --features RESNET18-TSM-ALL2 --ec_mode csi --time_span 30 --tau 0.9






## Requirements
Preliminary requirements:
- Python>=3.7
- CUDA 11.6

Run the following command if you have CUDA 11.6:
```
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyYAML, pandas, and [PyG](https://www.pyg.org)>=2.0.3 with CUDA>=11.1

## Installation
After confirming the above requirements, run the following commands:
```
git clone https://github.com/IntelLabs/GraVi-T.git
cd GraVi-T
pip3 install -e .
```

## Getting Started (Active Speaker Detection)
### Annotations
1) Download the annotations of AVA-ActiveSpeaker from the official site:
```
DATA_DIR="data/annotations"

wget https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2 -P ${DATA_DIR}
tar -xf ${DATA_DIR}/ava_activespeaker_val_v1.0.tar.bz2 -C ${DATA_DIR}
```

2) Preprocess the annotations:
```
python data/annotations/merge_ava_activespeaker.py
```

### Features
Download `RESNET18-TSM-AUG.zip` from the Google Drive link from [SPELL](https://github.com/SRA2/SPELL#code-usage) and unzip under `data/features`.
> We use the features from the thirdparty repositories.

### Directory Structure
The data directories should look as follows:
```
|-- data
    |-- annotations
        |-- ava_activespeaker_val_v1.0.csv
    |-- features
        |-- RESNET18-TSM-AUG
            |-- train
            |-- val
```

### Experiments
We can perform the experiments on active speaker detection with the default configuration by following the three steps below.

#### Step 1: Graph Generation
Run the following command to generate spatial-temporal graphs from the features:
```
python data/generate_spatial-temporal_graphs.py --features RESNET18-TSM-AUG --ec_mode csi --time_span 90 --tau 0.9
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video, which spans about 90 seconds (specified by `--time_span`).

#### Step 2: Training
Next, run the training script by passing the default configuration file:
```
python tools/train_context_reasoning.py --cfg configs/active-speaker-detection/ava_active-speaker/SPELL_default.yaml
```
The results and logs will be saved under `results`.

#### Step 3: Evaluation
Now, we can evaluate the trained model's performance:
```
python tools/evaluate.py --exp_name SPELL_ASD_default --eval_type AVA_ASD
```





