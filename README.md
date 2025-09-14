# PepperMint Role – SPELL Model

## Credit
This repository builds on the original **SPELL** model and codebase for active speaker detection developed by Min et al., available at [IntelLabs/GraVi-T](https://github.com/IntelLabs/GraVi-T).

Minor modifications have been made to adapt the code for specific use cases.

If you use this code or build upon it, please cite the original paper:

```
@inproceedings{min2022learning,
  title={Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  booktitle={European Conference on Computer Vision},
  pages={371--387},
  year={2022},
  organization={Springer}
}
```

---

## Feature Extraction

This repository contains code for training the graph neural network for the SPELL model. Training requires **visual and audio embeddings**.

### Option 1: Extract your own features
Visit the repository [active_speaker_encoder](https://github.com/BStephen99/active_speaker_encoder) and follow the instructions.

If you have trained your own encoder, you can build feature dictionaries using:

- `getFeatureDictionaryPepperMint.py`
- `getFeatureDictionaryAVA.py`
- `getFeatureDictionaryWASD.py`

### Option 2: Use preprocessed features
Download preprocessed feature dictionaries (from the ALL model) at [Ortolang: PepperMint](https://www.ortolang.fr/workspaces/peppermint?section=content&root=head&path=%2F).  
Unzip `RESNET18-TSM-ALL.zip` under `data/features`.

---

## Directory Structure
Your data directories should look like this:

```
|-- data
    |-- features
        |-- RESNET18-TSM-ALL
            |-- AVAtrain
            |-- test
            |-- train
            |-- WASDtrain
            |-- WASDval
```

---

## Setup

### Requirements
- Python >= 3.7
- CUDA 11.6

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Or manually install:
- PyYAML  
- pandas  
- [PyG](https://www.pyg.org) >= 2.0.3 with CUDA >= 11.1  

### Installation
```bash
git clone https://github.com/BStephen99/PepperMint.git
cd PepperMint
pip3 install -e .
```

---

## Training & Evaluation

### Modes
Available modes:
- `pepper` – PepperMint Role dataset
- `wasd` – WASD dataset
- `ava` – AVA dataset
- `addressee` – Addressee estimation (PepperMint Role)

Model variants include:
`SPELLBYPLAY`, `SPELLBYPLAYGAZE`, `SPELLBYPLAYLAND`,  
`SPELLBYPLAY2FEATS`, `SPELLBYPLAYAUDIOONLY`, `SPELLVISONLYBYPLAY`.

---

### Graph Generation
Generate spatial-temporal graphs from features:

```bash
# All datasets
python data/generate_spatial-temporal_graphs_allsets.py --features RESNET18-TSM-ALL --ec_mode csi --time_span 30 --tau 0.9

# PepperMint Role + WASD (with gender & landmarks)
python data/generate_spatial-temporal_graphs_landmarks_gender.py --features RESNET18-TSM-ALL --ec_mode csi --time_span 30 --tau 0.9

# PepperMint Role (with addressee annotations)
python data/generate_spatial-temporal_graphs_peppermint.py --features RESNET18-TSM-ALL --ec_mode csi --time_span 30 --tau 0.9
```

Graphs are saved in `data/graphs`, with each graph covering 30 seconds (`--time_span`).

---

### Training
Run training with a config file from `./configs/active-speaker-detection/ava_active-speaker`:

```bash
python tools/train_context_reasoning.py --cfg configs/active-speaker-detection/ava_active-speaker/SPELL_default.yaml

# Multiclass / Addressee detection
python tools/train_context_reasoning_multiclass.py --cfg configs/active-speaker-detection/ava_active-speaker/SPELL_Addressee.yaml
```

Results and logs are stored in `results/`.

---

### Evaluation
Evaluate trained models:

```bash
python tools/evaluate.py --exp_name <EXP_NAME> --eval_type AVA_ASD --mode <MODE>
```

Where `<MODE>` can be:
- `pepper` → ASD on PepperMint Role test set  
- `wasd` → ASD on WASD test set  
- `ava` → ASD on AVA val set  
- `addressee` → Addressee estimation on PepperMint Role test set  

