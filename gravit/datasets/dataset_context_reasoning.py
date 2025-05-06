import os
import glob
import torch
from torch_geometric.data import Data, Dataset


class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs, valSet=False):
        super(GraphDataset, self).__init__()
        print("path_graphs", path_graphs)
        print(valSet)
        if "train" in path_graphs: #or "val" in path_graphs:
            #self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
            print("train")
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/AVAtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/AVAtrain/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-WASD_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-OURS_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt") + glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt") )
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/train/*.pt"))
            self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-WASD_csi_30.0_0.9/val_AVA/*.pt"))
            #self.all_graphs = sorted(glob.glob('/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/WASDtrain/*'))
            #print("train ours")
        elif valSet:
            print("val")
            print(path_graphs)
            #self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
            self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/test/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDval/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-OURS_csi_30.0_0.9/test/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/test/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-WASD_csi_30.0_0.9/test/*.pt"))
            #self.all_graphs = sorted(glob.glob(os.path.join("./data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/val_WASD", '*.pt')))
            #self.all_graphs = sorted(glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/ours", '220927*.pt')))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/ours/220927*.pt"))
            #self.all_graphs = sorted(glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/ours", '220929*.pt'))+ glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/ours", '220928*.pt'))+ glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/ours", '220926*.pt')))
            #self.all_graphs = sorted(glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/AVAval", '*.pt'))+glob.glob(os.path.join("./data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/WASDval", '*.pt')))
            #self.all_graphs = sorted(glob.glob(os.path.join("./data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/ours", '220927*.pt')))
            #self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '220927*.pt')))
            #self.all_graphs = sorted(glob.glob('/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/WASDval/*'))
        #else:
        #    self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '220929*.pt')) + glob.glob(os.path.join(path_graphs, '220926*.pt')) + glob.glob(os.path.join(path_graphs, '220928*.pt')))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
            #print(self.all_graphs[idx])
            data = torch.load(self.all_graphs[idx])
            return data


            #/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/WASDtrain/KVBPf_PYmPE_92-122_0001.pt
