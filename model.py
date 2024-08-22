import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from sklearn.utils import shuffle
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool
import time
from torch_geometric.loader import DataLoader
from copy import deepcopy
from torch_geometric.data import Batch
import torch_geometric
import multiprocessing as mp
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from utils import *
from sklearn.model_selection import train_test_split
from collections import defaultdict
import itertools


def get_pair_list(indices):
    pair_dicts = []
    pair_dict = {}

    for _, r in tqdm(enumerate(indices), total=len(indices)):
        if _ % 5000 == 0:
            pair_dicts.append(pair_dict)
            pair_dict = {}
        core = smiles[r][1]
        cond = whole_conditions[r]
        if core not in list(pair_dict):
            pair_dict[core] = [cond]
        else:
            pair_dict[core].append(cond)

    pair_dicts.append(pair_dict)
    
    pair_dict = defaultdict(list)

    for d in pair_dicts:
        for key, value in d.items():
            pair_dict[key].append(value)
            
    for core in list(pair_dict):
        rows_rows = pair_dict[core]
        rows = []
        for r in rows_rows:
            rows = rows + r

        pair_dict[core] = rows
        
    return pair_dict


def make_graph_order(smiles, whole, scaff, cond, key=None):
    if key == None:
        graph_order, action_order, labels, masks, whole, paths = generate_graphs(whole, scaff)
        if cond != None:
            condition = torch.Tensor([cond])
        else:
            condition = torch.zeros(2)
    else:
        graph_order, action_order, labels, masks, whole, paths = generate_graphs(smiles[key][0], smiles[key][1])
        condition = torch.Tensor([whole_conditions[key] + scaffold_conditions[key]])

    if graph_order == None:
        return None

    action_order, graph_actions = action_order
    latest, scaffold, lengths = masks
    
    whole.condition = condition

    whole.node_actions = action_order
    whole.atom_types = labels[0]
    whole.append_bond_types = labels[1]
    whole.connect_bond_types = labels[2]

    whole.node_stack = np.cumsum([len(graph.x) for graph in graph_order]).sum() 
    whole.length = len(graph_actions)

    whole.latest = latest
    whole.scaffold = scaffold
    whole.lengths = lengths

    history_batch = []
    for j in range(1, len(graph_actions) + 1):
        history_batch = history_batch + list(range(0, j))
    seq_lengths = list(range(1, len(graph_actions) + 1))

    whole.history_batch = torch.Tensor(history_batch)
    whole.seq_lengths = torch.Tensor(seq_lengths)
    
    return graph_order, whole


class GraphData(Dataset):
    def __init__(self, smiles, conditions, indices, key=True):
        self.smiles = smiles
        self.conditions = conditions
        self.indices = indices
        self.key = key
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        
        if self.key:
            graph = make_graph_order(self.smiles, None, None, None, key=idx)
        else:
            whole, scaff = self.smiles[idx]
            cond = self.conditions[idx]
            graph = make_graph_order(self.smiles, whole, scaff, cond)
            
        samp_to_long(graph)
        return graph
    
    
class MoleculeConv(MessagePassing):
    def __init__(self, input_dim, condition_dim, output_dim):
        super().__init__(aggr='add')

        self.node_lin = nn.Linear((input_dim + condition_dim) * 5, output_dim, bias=False)
        
        self.reset_parameters()
        
    def forward(self, node_features, edge_indexes, condition=None):
        
        if condition != None:
            node_features = torch.cat([node_features, condition], dim=1)
        
        props = [node_features]
        
        for r in range(len(edge_indexes)):
            edge_index = edge_indexes[r]
            props.append(self.propagate(edge_index, x=node_features))
            
        x = self.node_lin(torch.cat(props, dim=1))
        
        return x
    
    
class LinearBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.lin = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        x = self.lin(x)
        return self.bn(x)
    
    
class ConvNet(nn.Module):
    def __init__(self, dims, hidden, condition_dim):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        
        for r in range(len(dims) - 1):
            if r != 0:
                self.bn_layers.append(nn.BatchNorm1d(dims[r], dims[r]))
            self.conv_layers.append(MoleculeConv(dims[r], 0, dims[r + 1]))
            self.cond_layers.append(nn.Linear(dims[r + 1] + condition_dim, dims[r + 1]))
                    
        self.bn_conv = nn.BatchNorm1d(np.sum(dims[1:]))
                    
        self.combine = nn.Sequential(
            nn.Linear(np.sum(dims[1:]) + condition_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU()
        )
        
    def forward(self, scaffold_x, adjs, conditions, node_batch):
        outputs = []
        
        for r in range(len(self.conv_layers)):
            layer = self.conv_layers[r]
            
            if r != 0:
                bn = self.bn_layers[r - 1]
                scaffold_x = F.relu(bn(scaffold_x))
            
            if conditions != None:
                scaffold_x = self.cond_layers[r](torch.cat([layer(scaffold_x, adjs), conditions[node_batch]], dim=-1))
            else:
                scaffold_x = layer(scaffold_x, adjs)
            
            outputs.append(scaffold_x)
            
        scaffold_x = F.relu(self.bn_conv(torch.cat(outputs, dim=-1)))
        if conditions != None:
            scaffold_x = torch.cat([scaffold_x, conditions[node_batch]], dim=-1)
        
        scaffold_x = self.combine(scaffold_x)
        
        return scaffold_x
    

class MolVAE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, condition_dim, rnn=False):
        super(MolVAE, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden_channels
        self.rnn = rnn
        
        self.atom_embed = nn.Embedding(node_dim * 5, 16)
        
        enc_dims = [16, 32, 64, 128, 256]
        conv_dims = [16, 32, 64, 128, 256, 256]
        self.encoder = ConvNet(enc_dims, [128, 128], condition_dim)
        self.convnet = ConvNet(conv_dims, [256, 512], 128 + condition_dim)
        
        self.mean = nn.Linear(128, 128)
        self.log = nn.Linear(128, 128)
        
        self.reduce = nn.Linear(512, 64)
        
        self.rnn = nn.GRU(64, 512, num_layers=3)
                
        self.append_connect1 = LinearBN(1024 + 512 + 128 + condition_dim, 128)
        self.append_connect2 = nn.Linear(128, node_dim * edge_dim + edge_dim)
        
        self.end = nn.Linear(512 + 512 + 128 + condition_dim, 1)
        
    def forward(self, whole_data, scaffold_datas, conditions):
        batch = scaffold_datas.batch
        latest_appended = whole_data.latest.eq(1)
        
        node_batch = []
        graph_batch = []
        for r in range(len(whole_data.node_stack)):
            node_batch.extend([r] * whole_data.node_stack[r])
            graph_batch.extend([r] * whole_data.length[r])
                            
        whole_x = whole_data.x + self.node_dim * 4
        whole_edge = whole_data.edge_features
        
        bond_index = [whole_data.edge_index[:, whole_edge.eq(r)] for r in range(4)]
        adjs = bond_index
        
        whole_x = self.atom_embed(whole_x.long())
        whole_x = self.encoder(whole_x, adjs, conditions, whole_data.batch)
        z = global_mean_pool(whole_x, whole_data.batch)
        
        z_embed, z_mean, z_log = self.variate(z)
        if conditions != None:
            z_embed = torch.cat([z_embed, conditions], dim=-1)
            
        scaffold_x = scaffold_datas.x
        scaffold_edge = scaffold_datas.edge_features
    
        bond_index = [scaffold_datas.edge_index[:, scaffold_edge.eq(r)] for r in range(4)]
        adjs = bond_index
        
        scaffold_x = torch.where(
            whole_data.scaffold.eq(1),
            # If the atom is inside the scaffold
            scaffold_x + self.node_dim * 1,
            # If the atom is an ordinary atom
            torch.where(
                whole_data.latest.eq(1),
                # If the atom is latest appended
                scaffold_x + self.node_dim * 2,
                torch.where(
                    whole_data.latest.eq(2),
                    scaffold_x + self.node_dim * 3,
                    # If the atom is not latest appended
                    scaffold_x)))
        
        scaffold_x = self.atom_embed(scaffold_x.long())
        
        scaffold_x = self.convnet(scaffold_x, adjs, z_embed, node_batch)
        graph_embed = global_mean_pool(scaffold_x, batch)
        
        if self.rnn is True:
            history_embed = self.reduce(graph_embed)
                
            graph_history = torch.split(history_embed[whole_data.history_batch], whole_data.seq_lengths.tolist())
        
            padded_history = pad_sequence(graph_history, batch_first=True)
            pack_padded_history = pack_padded_sequence(padded_history, whole_data.seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                        
            out, graph_history = self.rnn(pack_padded_history)
            graph_embed = torch.cat([graph_embed, graph_history[-1]], dim=-1)
        
        graph_embed = torch.cat([graph_embed, z_embed[graph_batch]], dim=-1)
        action_embed = torch.cat([scaffold_x, graph_embed[batch]], dim=-1)
        
        action_embed = F.relu(self.append_connect1(action_embed))
        append_connect = self.append_connect2(action_embed)
        p_end = self.end(graph_embed)
        
        append_connect, p_end = self.segment_softmax(append_connect, p_end, batch)
        
        p_append = append_connect[:, :self.node_dim * self.edge_dim]
        p_connect = append_connect[:, self.node_dim * self.edge_dim:]
        
        p_append = p_append.view(-1, self.node_dim, self.edge_dim)
        
        p_append, p_connect, p_end = torch.log(p_append + 1e-6), torch.log(p_connect + 1e-6), torch.log(p_end + 1e-6)
        
        node_actions = whole_data.node_actions
        atom_types = whole_data.atom_types
        append_bond_types = whole_data.append_bond_types
        connect_bond_types = whole_data.connect_bond_types
        
        p_append = p_append[torch.arange(len(p_append)).long(), atom_types, append_bond_types]
        p_append = torch.where(node_actions.eq(0), p_append, p_append.new_zeros(1))
        
        p_connect = p_connect[torch.arange(len(p_connect)).long(), connect_bond_types]
        p_connect = torch.where(node_actions.eq(1), p_connect, p_connect.new_zeros(1))
        
        p_end = torch.where(node_actions.eq(2), p_end[batch], p_end.new_zeros(1))
        
        p_total = p_append + p_connect + p_end
        p_total = global_add_pool(global_add_pool(p_total, batch), torch.Tensor(graph_batch).long().to(device))
        
        vae_loss = -0.5 * torch.sum(1 + z_log - z_mean.pow(2) - z_log.exp(), dim=1)
        
        return -p_total.mean(), vae_loss.mean()
    
    def generate(self, scaffold_datas, conditions):
        scaffold_datas = deepcopy(scaffold_datas)
        for data in scaffold_datas:
            data.scaffold = torch.ones(len(data.x), device=device)
            data.latest = torch.zeros(len(data.x), device=device)
        
        z_embed = torch.randn(len(scaffold_datas), 128, device=device)
        if conditions != None:
            z_embed = torch.cat([z_embed, conditions], dim=-1)
        add_more = [True] * len(scaffold_datas)
        
        history = [[] for x in range(len(scaffold_datas))]
        
        max_steps = 100
        
        for r in range(max_steps):
            if all([x == False for x in add_more]):
                break
            
            scaffold_batch = Batch.from_data_list(list(itertools.compress(scaffold_datas, add_more)))
            batch = scaffold_batch.batch
            
            scaffold_x = scaffold_batch.x
            scaffold_edge = scaffold_batch.edge_features
            
            scaffold_x = torch.where(
            scaffold_batch.scaffold.eq(1),
            scaffold_x + self.node_dim * 1,
            torch.where(
                scaffold_batch.latest.eq(1),
                scaffold_x + self.node_dim * 2,
                torch.where(
                    scaffold_batch.latest.eq(2),
                    scaffold_x + self.node_dim * 3,
                    scaffold_x)))
            
            bond_index = [scaffold_batch.edge_index[:, scaffold_edge.eq(r)] for r in range(4)]
            adjs = bond_index # + dist_index
        
            scaffold_x = self.atom_embed(scaffold_x)

            scaffold_x = self.convnet(scaffold_x, adjs, z_embed, batch)
            graph_embed = global_mean_pool(scaffold_x, batch)
            snapshot = self.reduce(graph_embed)
            
            vect_index = 0
            for r in range(len(add_more)):
                if add_more[r] is False:
                    continue
                history[r].append(snapshot[vect_index])
                vect_index += 1
                
            if self.rnn is True:
                graph_history = [torch.stack(x, dim=0) for idx, x in enumerate(history) if add_more[idx] is True]
                seq_lengths = [len(x) for x in graph_history]
            
                padded_history = pad_sequence(graph_history, batch_first=True)
                pack_padded_history = pack_padded_sequence(padded_history, seq_lengths, batch_first=True, enforce_sorted=False)
                
                out, graph_history = self.rnn(pack_padded_history)
                graph_embed = torch.cat([graph_embed, graph_history[-1]], dim=-1)
            
            graph_embed = torch.cat([graph_embed, z_embed[add_more]], dim=-1)
            action_embed = torch.cat([scaffold_x, graph_embed[batch]], dim=-1)

            action_embed = F.relu(self.append_connect1(action_embed))
            append_connect = self.append_connect2(action_embed)
            p_end = self.end(graph_embed)
            
            p_append_connect, p_end = self.segment_softmax(append_connect, p_end, batch)
            p_append_connects = [p_append_connect[batch == x] for x in range(len(batch.unique()))]
                
            max_values = [x.max() for x in p_append_connects]
            current_mask = []

            vect_index = 0
            for r in range(len(add_more)):
                if add_more[r] is False:
                    continue
                    
                if max_values[vect_index] < p_end[vect_index]:
                    add_more[r] = False
                    current_mask.append(False)
                else:
                    current_mask.append(True)
                vect_index += 1
            
            max_values = list(itertools.compress(max_values, current_mask))
            p_append_connects = list(itertools.compress(p_append_connects, current_mask))
            
            atom_indices = []
            action_indices = []
            
            for r in range(len(p_append_connects)):
                atom_index, action_index = (p_append_connects[r] == max_values[r]).nonzero()[0]
                atom_indices.append(atom_index)
                action_indices.append(action_index)
                
            p_actions = [p_append_connects[x][atom_indices[x]] for x in range(len(atom_indices))]
            vect_index = 0
                        
            for r in range(len(add_more)):
                if add_more[r] == False:
                    continue
                    
                p_action = p_actions[vect_index]
                action_index = action_indices[vect_index]
                atom_index = atom_indices[vect_index]
                scaffold_data = scaffold_datas[r]
                max_value = max_values[vect_index]
                
                if action_index < self.node_dim * self.edge_dim:
                    p_action = p_action[:self.edge_dim * self.node_dim].view(-1, self.node_dim, self.edge_dim)
                    _, atom_type, bond_type = (p_action == max_value).nonzero()[0]

                    atom_type = atom_type.unsqueeze(0)
                    bond_type = bond_type.unsqueeze(0)

                    scaffold_data.x = torch.cat([scaffold_data.x, atom_type], dim=0)
                    scaffold_data.edge_index = torch.cat([scaffold_data.edge_index, torch.Tensor([[len(scaffold_data.x) - 1, atom_index], [atom_index, len(scaffold_data.x) - 1]]).to(device).long()], dim=1)
                    scaffold_data.edge_features = torch.cat([scaffold_data.edge_features, bond_type.repeat(2)], dim=0)
                    scaffold_data.latest = torch.cat([bond_type.new_zeros(len(scaffold_data.latest)), bond_type.new_ones(1)])
                    scaffold_data.scaffold = torch.cat([scaffold_data.scaffold, bond_type.new_zeros(1)])
                else:
                    p_action = p_action[self.edge_dim * self.node_dim:].view(self.edge_dim)

                    bond_type = (p_action == max_value).nonzero()[0]

                    scaffold_data.edge_index = torch.cat([scaffold_data.edge_index, torch.Tensor([[len(scaffold_data.x) - 1, atom_index], [atom_index, len(scaffold_data.x) - 1]]).to(device).long()], dim=1)
                    scaffold_data.edge_features = torch.cat([scaffold_data.edge_features, bond_type.repeat(2)], dim=0)
                    scaffold_data.latest[atom_index] = 2
                    
                vect_index += 1
            
        return scaffold_datas
    
    
    def variate(self, z):
        z_mean = self.mean(z)
        z_log = self.log(z)
        z_std = torch.exp(0.5 * z_log)
        norm = torch.randn(z_std.size(), device=device)
        z_whole = norm.mul(z_std).add_(z_mean)
        return z_whole, z_mean, z_log
        
        
    def segment_softmax(self, probs, term, batch):
        vector = torch.cat([probs, term[batch]], dim=-1)
        
        atom_max, _ = torch.max(vector, dim=-1)
        atom_max = global_max_pool(atom_max, batch)
        
        vector = vector[:, :-1] - atom_max[batch].unsqueeze(dim=1)
        term = term.squeeze(dim=1) - atom_max
        
        vector_exp, term_exp = torch.exp(vector), torch.exp(term)
        vector_sum = global_add_pool(vector_exp.sum(dim=-1), batch)
     
        term_sum = vector_sum + term_exp + 1e-6
        vector_softmax = vector_exp / term_sum[batch].unsqueeze(dim=1)
        term_softmax = term_exp / term_sum
    
        return vector_softmax, term_softmax
    
    
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L  
