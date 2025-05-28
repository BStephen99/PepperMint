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
  
        elif test_sets:
            self.all_graphs = []

            for t in test_sets:
                print(os.path.join(path_graphs, t, '*.pt'))
                self.all_graphs += glob.glob(os.path.join(path_graphs, t, '*.pt'))
            self.all_graphs = sorted(self.all_graphs)

 

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
            #print(self.all_graphs[idx])
            data = torch.load(self.all_graphs[idx])
            return data

