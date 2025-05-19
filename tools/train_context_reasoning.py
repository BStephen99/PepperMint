import os
import yaml
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model, get_loss_func
from gravit.datasets import GraphDataset
from torch.nn import CrossEntropyLoss


def invert_bbox(center_x, center_y, width, height):
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def train(cfg):
    """
    Run the training process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    training_sets = cfg["training_sets"]
    test_sets = cfg["test_sets"]

    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')
    os.makedirs(path_result, exist_ok=True)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(cfg['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'cfg.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)

    """
    state_dict = torch.load('/home2/bstephenson/GraVi-T/results/SPELL_ASD_byplayGaze/ckpt_best1.pt', map_location=torch.device(device))
    #state_dict = torch.load('/home2/bstephenson/GraVi-T/results/SPELL_ASD_defaultPepperSpeak4/ckpt_bestlandmarksAndSpeakerEmb.pt', map_location=torch.device(device))
    model_state_dict = model.state_dict()
    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)
    #model.load_state_dict(state_dict)
    print("Loaded pretrained weights partially")
    model.to(device)
    """
    


    train_loader1 = DataLoader(GraphDataset(path_graphs, training_sets=training_sets), batch_size=cfg['batch_size'], shuffle=True)
    #train_loader1 = DataLoader(GraphDataset("./data/graphs/RESNET18-TSM-AUG4_csi_90.0_0.9/train"), batch_size=cfg['batch_size'], shuffle=True)
    #train_loader2 = DataLoader(GraphDataset(os.path.join(path_graphs, 'ours')), batch_size=cfg['batch_size'], shuffle=True)
    #combined_train = DataLoader(train_loader1.dataset + train_loader2.dataset, batch_size=2)
    combined_train = train_loader1

    val_loader = DataLoader(GraphDataset(path_graphs, test_sets=test_sets))

    # Prepare the experiment
    loss_func = get_loss_func(cfg) 
    print(loss_func)
    #genderLoss = CrossEntropyLoss()
    loss_func_val = get_loss_func(cfg, 'val')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['sch_param'])

    # Run the training process
    logger.info('Training process started')

    min_loss_val = float('inf')
    for epoch in range(1, cfg['num_epoch']+1):
        model.train()

        # Train for a single epoch
        loss_sum = 0
        #for data in train_loader:
        for data in combined_train:
            #print(data)
            optimizer.zero_grad()

            #train ASD or addressee estimation
            data.y[data.y == 2] = 1 #using
            data.y[data.y == 3] = 1
            #data.y[data.y == 2] = 0
            #x, y = data.x.to(device), data.y.to(device) #using
            x, y = data.x.to(device), data.y.squeeze(dim=1).to(device)
            #xH = data.xH.to(device)
            #y = y.long()

            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if cfg['use_spf']:
                #print(data.c.shape)
                c = data.c.to(device)
                #cH = data.ch.to(device)
                try:
                    ps = data.ps.to(device)
                    pers = data.perSpeak.to(device)
                    speakerEmb = data.speakerEmb.to(device)
                    gender = data.gender.to(device)
                    
                    if "landmarks" in data:
                        landmarks = data.landmarks.to(device) 
                    else: 
                        k=99
                        landmarks = data.landmarks_back.to(device)
                        #landmarksHigh = data.landmarks_high.to(device)
                    
                    #landmarksHigh = data.landmarks_high.to(device)
                    #numPredSpeakers = data.numPredSpeakers.to(device)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("********************************************")
                    ps = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    pers = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    #gender = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    #landmarks = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    #landmarksHigh = torch.tensor([0]*cH.shape[0], dtype=torch.float32).unsqueeze(1).to(device)
                    numPredSpeakers = None



            #logits = model(x, edge_index, edge_attr, c, ps, pers)
            #logits = model(x, edge_index, edge_attr, c, ps, dinoEmb=dinoEmb, speakerEmb=speakerEmb)
            #logits = model(x, xH, edge_index, edge_attr, c, cH, ps,pers=pers, gender=gender, landmarks=landmarks, landmarksH=landmarksHigh, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, xH=xH, c=c, cH=cH, ps=ps,pers=pers, gaze=gaze, gender=gender, landmarks=landmarks, landmarksH=landmarksHigh, speakerEmb=speakerEmb)
            logits = model(x, edge_index, edge_attr, xH=None, c=c, cH=None, ps=ps,pers=pers, gaze=None, gender=gender, landmarks=landmarks, landmarksH=None, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, c, ps, gender=gender, gaze=gaze, landmarks=landmarks, speakerEmb=speakerEmb, numPredSpeakers=numPredSpeakers)
            #logits = model(x, edge_index, edge_attr, c, ps, pers=pers, gender=gender, gaze=gaze, landmarks=landmarks, speakerEmb=speakerEmb, numPredSpeakers=numPredSpeakers, bodyEmb=bodyEmb)

        
            #loss = loss_func(logits, y) #+ genderLoss(predictGender, gender.view(-1))
            loss = loss_func(logits.squeeze(), y)
            #loss = loss_func(logits, y.long())
            #print("loss", loss)
            loss.backward()
            #print(loss)
            loss_sum += loss.item()
            optimizer.step()

        # Adjust the learning rate
        scheduler.step()

        loss_train = loss_sum / len(combined_train) #len(train_loader)

        # Get the validation loss
        loss_val = val(val_loader, cfg['use_spf'], model, device, loss_func_val)
        #print(loss_val)

        # Save the best-performing checkpoint

        if loss_val < min_loss_val:
            min_loss_val = loss_val
            epoch_best = epoch
            #torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))
        torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_last.pt'))

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, best: epoch {epoch_best:03d}')

    logger.info('Training finished')


def val(val_loader, use_spf, model, device, loss_func):
    """
    Run a single validation process
    """

    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for data in val_loader:
            data.y[data.y == 2] = 1
            data.y[data.y == 3] = 1
            #data.y[data.y == 2] = 0
            #x, y = data.x.to(device), data.y.to(device)
            x, y = data.x.to(device), data.y.squeeze(dim=1).to(device)
            #xH = data.xH.to(device)
            #y = y.long()
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if use_spf:
                c = data.c.to(device)
                #cH = data.ch.to(device)
                ps = data.ps.to(device)
                pers = data.perSpeak.to(device)
                speakerEmb = data.speakerEmb.to(device)

                
                gender = data.gender.to(device)
                if "landmarks" in data:
                        landmarks = data.landmarks.to(device) 
                else: 
                        #k = 99
                        landmarks = data.landmarks_back.to(device)
                        #landmarksHigh = data.landmarks_high.to(device)
                
                #landmarks = data.landmarks_back.to(device)
                #landmarksHigh = data.landmarks_high.to(device)
                #numPredSpeakers = data.numPredSpeakers.to(device)
                #gender = torch.tensor([0]*c.shape[0], dtype=torch.float32).unsqueeze(1).to(device)



            #logits = model(x, edge_index, edge_attr, c, ps, pers)
            #logits = model(x, edge_index, edge_attr, c, ps)
            #logits = model(x, edge_index, edge_attr, c, ps,dinoEmb=dinoEmb, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, c, ps, gender=gender,gaze=gaze, landmarks=landmarks, speakerEmb=speakerEmb, numPredSpeakers=numPredSpeakers)
            #logits = model(x, edge_index, edge_attr, c, ps, gender=gender,gaze=None, landmarks=None, speakerEmb=speakerEmb, numPredSpeakers=None)
            #logits = model(x, edge_index, edge_attr, c, ps, pers=pers, gender=gender, landmarks=landmarks, speakerEmb=speakerEmb)
            #logits = model(x, xH, edge_index, edge_attr, c, cH, ps,pers=pers, gender=gender, landmarks=landmarks, landmarksH=landmarksHigh, speakerEmb=speakerEmb)
            logits = model(x, edge_index, edge_attr, xH=None, c=c, ps=ps,pers=pers, gender=gender, gaze=None, landmarks=landmarks, landmarksH=None, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, xH=xH, c=c, cH=cH, ps=ps,pers=pers, gaze=gaze, gender=gender, landmarks=landmarks, landmarksH=landmarksHigh, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, xH=None, c=c, cH=None, ps=ps,pers=pers, gaze=None, gender=None, landmarks=None, landmarksH=None, speakerEmb=speakerEmb)
            #logits = model(x, edge_index, edge_attr, c, ps, gender=gender, gaze=gaze, landmarks=landmarks, speakerEmb=speakerEmb, numPredSpeakers=numPredSpeakers)
            #logits = model(x, edge_index, edge_attr, c, ps, pers=pers, gender=gender, gaze=gaze, landmarks=landmarks, speakerEmb=speakerEmb, numPredSpeakers=numPredSpeakers, bodyEmb=bodyEmb)
            
            #print(logits)
            loss = loss_func(logits.squeeze(), y)

            #loss = loss_func(logits, y.long())
            loss_sum += loss.item()

    return loss_sum / len(val_loader)


if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)

    train(cfg)
