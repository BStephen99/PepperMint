import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, MultiheadAttention
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm
from torch.nn import Embedding

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

class SPELL(Module):
    def __init__(self, cfg):
        super(SPELL, self).__init__()
        self.use_spf = cfg['use_spf']
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']


        self.self_attn = MultiheadAttention(embed_dim=channels[0], num_heads=1, dropout=dropout, batch_first=True)
    
        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim'])
            self.layer_gaze = Linear(-1, cfg['proj_dim'])
            self.layer_pose = Linear(-1, cfg['proj_dim'])
            self.layer_ps = Linear(-1, cfg['proj_dim'])
            self.layer_speakerEmb = Linear(-1, 10)
            self.layer_gender = Embedding(3, 20)
            self.layer_identity = Linear(-1, channels[0])

        self.attention = MultiheadAttention(embed_dim=640, num_heads=1, dropout=dropout, batch_first=True)
        self.output_proj = Linear(640, channels[0])

        
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

        self.final_proj = Linear(channels[1], 1)



    def forward(self, x, edge_index, edge_attr, c=None, ps=None, pers=None, gender=None, gaze=None, landmarks=None, dinoEmb=None, speakerEmb=None, numPredSpeakers=None):
        feature_dim = x.shape[1]
        if self.use_spf:
            query = key = value = torch.cat((x[:, feature_dim//self.num_modality:], self.layer_pose(landmarks), self.layer_spf(c)), dim=1)
            x_visual, _ = self.attention(query, key, value)
            #x_visual = self.attention(torch.cat((x[:, feature_dim//self.num_modality:], self.layer_pose(landmarks), self.layer_spf(c)), dim=1))
        else:
            x_visual = self.attention(x[:, :feature_dim//self.num_modality])

        """
        if self.num_modality == 2:
            query = key = value = torch.cat((x[:, :feature_dim//self.num_modality], self.layer_speakerEmb(speakerEmb), ps, gender), dim=1)
            x_audio, _ = self.self_attn(query, key, value)
            x = x_visual + x_audio
        else:
        """
        x = x_visual
        x = self.output_proj(x)
        #print(x.shape)
        
        x = self.batch01(x)
        x = self.relu(x)
        
        #x, _ = self.self_attn(x, x, x)
        
        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)


        out = x1 + x2 + x3

        out = self.final_proj(out)

        return out