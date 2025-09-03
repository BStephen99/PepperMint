import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="Generate experiment config YAML")

    # Allow overriding some key values via CLI
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model_name", default="SPELL", required=True)
    parser.add_argument("--graph_name", default="RESNET18-TSM-ALL_csi_30.0_0.9", required=True)
    parser.add_argument("--loss_name", choices=["bce_logit", "ce"],default="bce_logit")  
    parser.add_argument("--out", default="configs/", help="Output folder for configs")

     # Example of list input
    parser.add_argument("--training_sets", nargs="+", default=["train"])
    parser.add_argument("--test_sets", nargs="+", default=["test"])
    parser.add_argument("--positiveLabels", nargs="+", default=["speaking_to_pepper", "speaking_to_human"])
    parser.add_argument("--positiveLabelsWASD", nargs="+", default=["SPEAKING_AUDIBLE"])
    parser.add_argument("--positiveLabelsAVA", nargs="+", default=["SPEAKING_AUDIBLE", "SPEAKING_INAUDIBLE"])

    # Other params (numbers, bools, paths, etc.)
    parser.add_argument("--graph_name", type=str, default="RESNET18-TSM-ALL_csi_30.0_0.9")
    parser.add_argument("--use_spf", action="store_true", default=True)
    parser.add_argument("--use_ref", action="store_true", default=False)
    parser.add_argument("--num_modality", type=int, default=2)
    parser.add_argument("--channel1", type=int, default=64)
    parser.add_argument("--channel2", type=int, default=16)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--final_dim", type=int, default=1)
    parser.add_argument("--num_att_heads", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sch_param", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=70)

    parser.add_argument("--csv_path", type=str, default="/home/brooke/Downloads/annotations.csv")
    parser.add_argument("--wasd_path", type=str, default="/media/brooke/PPM_Brooke/Mine/WASD/csv/val_orig_gender_landmarks_speaker_emb_corrected.csv")
    parser.add_argument("--ava_path", type=str, default="/media/brooke/PPM_Brooke/Mine/WASD/csv/val_orig_gender_landmarks_speaker_emb_corrected.csv")

    parser.add_argument("--multiclass", action="store_true", default=False)
    parser.add_argument("--gender", action="store_true", default=False)
    parser.add_argument("--gaze", action="store_true", default=False)
    parser.add_argument("--numPredSpeakers", action="store_true", default=False)
    parser.add_argument("--twoView", action="store_true", default=False)
    parser.add_argument("--laugh_not_speech", action="store_true", default=False)
    parser.add_argument("--landmarks", action="store_true", default=False)

    parser.add_argument("--root_data", type=str, default="./data")
    parser.add_argument("--root_result", type=str, default="./results")
    parser.add_argument("--output", type=str, default="config.yaml")

    args = parser.parse_args()

    # Full default config (your template)
    config = {
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "graph_name": args.graph_name,
        "loss_name": args.loss_name,
        "use_spf": True,
        "use_ref": False,
        "num_modality": 2,
        "channel1": 64,
        "channel2": 16,
        "proj_dim": 64,
        "final_dim": 1,
        "num_att_heads": 0,
        "dropout": 0.3,
        "lr": 0.0005,
        "wd": 0,
        "batch_size": 4,
        "sch_param": 10,
        "num_epoch": 70,
        "training_sets": ["train"],
        "test_sets": ["test"],
        "csv_path": "/home/brooke/Downloads/annotations.csv",
        "wasd_path": "/media/brooke/PPM_Brooke/Mine/WASD/csv/val_orig_gender_landmarks_speaker_emb_corrected.csv",
        "multiclass": False,
        "positiveLabels": ["speaking_to_pepper", "speaking_to_human"],
        "positiveLabelsWASD": ["SPEAKING_AUDIBLE"],
        "gender": False,
        "gaze": False,
        "numPredSpeakers": False,
        "twoView": False,
        "laugh_not_speech": False,
        "landmarks": False,
        "root_data": "./data",
        "root_result": "./results",
    }

    # Ensure output folder exists
    os.makedirs(args.out, exist_ok=True)

    # Save YAML
    out_path = os.path.join(args.out, f"{args.exp_name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ Config saved to {out_path}")

if __name__ == "__main__":
    main()
