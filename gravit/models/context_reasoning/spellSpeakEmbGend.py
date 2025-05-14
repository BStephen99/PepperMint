import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm
from torch.nn import Embedding
import torch.nn as nn
import torch.nn.functional as F


class GenderClassifier(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_classes=4):
        super(GenderClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

genClass = GenderClassifier()
genClass.load_state_dict(torch.load('gender_model.pt'))
genClass.eval()
genClass.to("cuda")


def nose_normalization(tensor, xnose_index=4, ynose_index=5):
    """
    Normalize body landmarks relative to the nose, preserving original zeros.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, 34).
        xnose_index (int): Index of the x-coordinate of the nose.
        ynose_index (int): Index of the y-coordinate of the nose.
    
    Returns:
        torch.Tensor: Normalized tensor of the same shape.
    """
    # Create a mask for original zeros
    zero_mask = (tensor == 0)
    
    # Extract xnose and ynose coordinates (batch_size,)
    xnose = tensor[:, xnose_index]
    ynose = tensor[:, ynose_index]
    
    # Reshape xnose and ynose to (batch_size, 1) for broadcasting
    xnose = xnose.unsqueeze(1)
    ynose = ynose.unsqueeze(1)
    
    # Subtract xnose from all x coordinates and ynose from all y coordinates
    # x coordinates are at even indices (0, 2, 4, ...), y coordinates at odd indices (1, 3, 5, ...)
    tensor[:, 0::2] -= xnose  # Normalize x coordinates
    tensor[:, 1::2] -= ynose  # Normalize y coordinates
    
    # Restore original zeros using the mask
    tensor[zero_mask] = 0
    
    return tensor


class SpeakerPredictor(torch.nn.Module):
    def __init__(self, feature_dim=1024):
        super(SpeakerPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim + 4, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 4)  # Output speaker bbox coordinates
        self.relu = torch.nn.ReLU()

    def forward(self, listener_feature, listener_box):
        x = torch.cat((listener_feature, listener_box), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

coorPred = SpeakerPredictor()
coorPred.load_state_dict(torch.load("speaker_coord_model.pth"))
coorPred.eval()


class LaughClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.0):
        super(LaughClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out2 = self.relu(out)
        out = self.dropout(out2)  # Apply dropout
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out, out2

laugh = LaughClassifier(512,128,1)
laugh.load_state_dict(torch.load("laugh_weights.pth"))
laugh.eval()




class DilatedResidualLayer(Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# This is for the iterative refinement (we refer to MSTCN++: https://github.com/sj-li/MS-TCN2)
class Refinement(Module):
    def __init__(self, final_dim, num_layers=10, interm_dim=64):
        super(Refinement, self).__init__()
        self.conv_1x1 = Conv1d(final_dim, interm_dim, kernel_size=1)
        self.layers = ModuleList([DilatedResidualLayer(2**i, interm_dim, interm_dim) for i in range(num_layers)])
        self.conv_out = Conv1d(interm_dim, final_dim, kernel_size=1)

    def forward(self, x):
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        out = self.conv_out(f)
        return out


class SPELLSPEAKEMB(Module):
    def __init__(self, cfg):
        super(SPELLSPEAKEMB, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features
            self.layer_pose = Linear(-1, cfg['proj_dim'])
            self.layer_speakerEmb = Linear(-1, 10)
            self.layer_gender = Embedding(3, 5)
            self.speakerNorm = BatchNorm(192)
            self.coorPred = coorPred
            self.LaughClassifier = laugh
            self.laughNorm = BatchNorm(128)
            self.visualNorm = BatchNorm(cfg['proj_dim'])
            self.audioNorm = BatchNorm(cfg['proj_dim'])

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            self.layer21 = SAGEConv(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    #def forward(self, x, edge_index, edge_attr, c=None):
    def forward(self, x, edge_index, edge_attr, xH=None, c=None, cH=None, ps=None, pers=None, gender=None, gaze=None, landmarks=None, landmarksH=None, dinoEmb=None,speakerEmb=None, numPredSpeakers=None):
        feature_dim = x.shape[1]

        gender = self.layer_gender(gender.long()).squeeze(1)

        #cpred = self.coorPred(x, c)

        #with torch.no_grad():  # Disable gradient calculation
        #    laugh, laughEmb = self.LaughClassifier(x[:, feature_dim//self.num_modality:])
        #    laughEmb = self.laughNorm(laughEmb)
        #print(laughEmb[0])
        #print(cpred)

        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, feature_dim//self.num_modality:], self.layer_spf(c)), dim=1))
    
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
    
            #x_audio = self.layer012(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_speakerEmb(speakerEmb), ps, gender), dim=1))
            x_audio = self.layer012(torch.cat((x[:, :feature_dim//self.num_modality], self.speakerNorm(speakerEmb), ps,  gender), dim=1))
            #x_audio = self.layer012(torch.cat((x[:, :feature_dim//self.num_modality], F.softmax(genClass(speakerEmb), dim=1), ps,  gender), dim=1))
            #x_audio = self.layer012(torch.cat((x[:, :feature_dim//self.num_modality], self.speakerNorm(speakerEmb), ps), dim=1))

            x_visual = self.visualNorm(x_visual)
            x_audio = self.audioNorm(x_audio)

            x = x_visual + x_audio
            #x = self.dropoutEmb(x)
            #x = x_audio

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        out = x1+x2+x3
        #print(out)


        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr2), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out
