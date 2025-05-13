import os
import glob
import torch
from torch_geometric.data import Data, Dataset


class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs, training_sets=None, test_sets=None):
        super(GraphDataset, self).__init__()

        print("path_graphs", path_graphs)
        if training_sets:
            self.all_graphs = []

            for t in training_sets:
                print(os.path.join(path_graphs, t, '*.pt'))
                self.all_graphs += glob.glob(os.path.join(path_graphs, t, '*.pt'))
            self.all_graphs = sorted(self.all_graphs)
            #self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))

        
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/AVAtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt")+glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/AVAtrain/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-WASD_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-OURS_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt") + glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/WASDtrain/*.pt") )
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-AVA_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/train/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-LAUGH_csi_30.0_0.9/WASDtrain/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-WASD_csi_30.0_0.9/val_AVA/*.pt"))
            #self.all_graphs = sorted(glob.glob('/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL_csi_30.0_0.9/WASDtrain/*'))
            #print("train ours")
        elif test_sets:
            self.all_graphs = []

            for t in test_sets:
                print(os.path.join(path_graphs, t, '*.pt'))
                self.all_graphs += glob.glob(os.path.join(path_graphs, t, '*.pt'))
            self.all_graphs = sorted(self.all_graphs)

            #self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-ALL2_csi_30.0_0.9/test/*.pt"))
            #self.all_graphs = sorted(glob.glob("/home2/bstephenson/GraVi-T/data/graphs/RESNET18-TSM-LAUGH_csi_30.0_0.9/WASDval/*.pt"))
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
