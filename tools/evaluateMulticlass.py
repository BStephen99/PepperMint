import os
import glob
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model
from gravit.datasets import GraphDataset

#from gravit.utils.eval_tool import get_eval_score
#from gravit.utils.vs import avg_splits

from gravit.utils import eval_byplay as byplay
from gravit.utils import eval_tool as pepper
from gravit.utils import eval_WASD as wasd
from gravit.utils import formatter_pepper
from gravit.utils import formatter_WASD
from gravit.utils import formatter_multiclass


def select_eval_tool(mode="pepper"):
    if mode == "byplay":
        return byplay.get_eval_score, formatter_Multiclass.get_formatting_data_dict, formatter_Multiclass.get_formatted_preds
    elif mode == "wasd":
        return wasd.get_eval_score, formatter_WASD.get_formatting_data_dict, formatter_WASD.get_formatted_preds
    elif mode == "pepper":
        return pepper.get_eval_score, formatter_pepper.get_formatting_data_dict, formatter_pepper.get_formatted_preds
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    test_sets = cfg["test_sets"]


    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])
    logger.info(path_result)
    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = "cpu"#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)

    val_loader = DataLoader(GraphDataset(path_graphs, test_sets))
    num_val_graphs = len(val_loader)
    print("num val graphs",num_val_graphs)


    # Load the trained model
    logger.info('Loading the trained model')
    state_dict = torch.load(os.path.join(path_result, 'ckpt_best.pt'), map_location=torch.device('cpu'))
    #state_dict = torch.load(os.path.join(path_result, 'ckpt_bestAVA.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Load the feature files to properly format the evaluation results
    logger.info('Retrieving the formatting dictionary')
    data_dict = get_formatting_data_dict(cfg)

    # Run the evaluation process
    logger.info('Evaluation process started')

    preds_all = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 1):
            g = data.g.tolist()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if cfg['use_spf']:
                try:
                    c = data.c.to(device)
                    ps = data.ps.to(device)
                    pers = data.perSpeak.to(device)
                    speakerEmb=data.speakerEmb.to(device)
                except:
                    c = data.c.to(device)
                    ps = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    pers = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)

     
            logits = model(x, edge_index, edge_attr, xH=None, c=c, ps=ps,pers=pers, gender=None, gaze=None, landmarks=None, landmarksH=None, speakerEmb=speakerEmb)


            # Change the format of the model output
            preds = get_formatted_preds(cfg, logits, g, data_dict)
            preds_all.extend(preds)

            logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')


    # Compute the evaluation score
    logger.info(f'Computing the evaluation score')
    eval_score = get_eval_score(cfg, preds_all)
    logger.info(f'{cfg["eval_type"]} evaluation finished: {eval_score}\n')
    return eval_score

if __name__ == "__main__":
    """
    Evaluate the trained model from the experiment "exp_name"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset')
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment', required=True)
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation', required=True)
    parser.add_argument('--split',         type=int,   help='Split to evaluate')
    parser.add_argument('--all_splits',    action='store_true',   help='Evaluate all splits')
    parser.add_argument('--modelNum',      type=str,  default="None", help='Name of model')
    parser.add_argument('--mode',          type=str,  default="pepper", help='Evaluation mode')
    parser.add_argument('--gaze',          type=bool,  default=False, help='Gaze used')
    parser.add_argument('--twoView',       type=bool,  default=False, help='Two Views')

    args = parser.parse_args()

    get_eval_score, get_formatting_data_dict, get_formatted_preds = select_eval_tool(mode=args.mode)

    path_result = os.path.join(args.root_result, args.exp_name)
    if not os.path.isdir(path_result):
        raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

    results = []
    if args.all_splits:
        results = glob.glob(os.path.join(path_result, "*", "cfg.yaml"))
    else:
        if args.split:
            path_result = os.path.join(path_result, f'split{args.split}')
            if not os.path.isdir(path_result):
                raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

        results.append(os.path.join(path_result, 'cfg.yaml'))

    all_eval_results = []
    for result in results:
        args.cfg = result
        cfg = get_cfg(args)
        all_eval_results.append(evaluate(cfg))

    if "VS" in args.eval_type and args.all_splits:
        avg_splits.print_results(all_eval_results)
