from rdkit import Chem
from torch_geometric.data import Data
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch_geometric.data.data import BaseData
from torch_geometric.data import Batch
import torch_geometric
from collections.abc import Mapping
from typing import List, Optional, Sequence, Union
from torch.utils.data import Dataset
import random
from copy import copy
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.Recap import RecapDecompose

bond_list = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

a_types = pd.read_csv('datasets/atom_types.txt', header=None)
atom_list = []

for r in range(len(a_types)):
    atom_list.append((a_types.iloc[r, 0], a_types.iloc[r, 1], a_types.iloc[r, 2]))
    

def make_graph(smiles):
    edge_index = []
    edge_dict = {}
    edge_features = []
    node_features = []

    molecule = Chem.MolFromSmiles(smiles)

    atoms = list(molecule.GetAtoms())
    edge_count = 0
    
    for row in range(len(atoms)):
        atom = atoms[row]
        try:
            atom_f = atom_list.index((atom.GetSymbol(), atom.GetFormalCharge(), atom.GetNumExplicitHs()))
        except ValueError:
            return None
        if atom_f == None:
            return None
        node_features.append(atom_f)
        
    for bond in molecule.GetBonds():
        row = bond.GetBeginAtomIdx()
        col = bond.GetEndAtomIdx()

        bond_f = bond_list.index(bond.GetBondType())
        
        edge_features.append(bond_f)
        edge_features.append(bond_f)
        edge_index.append([row, col])
        edge_index.append([col, row])
        
        edge_count += 2
                
    node_features = torch.Tensor(node_features)
    edge_features = torch.Tensor(edge_features)
    edge_index = torch.Tensor(edge_index).transpose(0, 1)
    return Data(x=node_features, edge_index=edge_index, edge_features=edge_features), molecule


def make_analog_pair(whole, scaffold):
    whole_data, whole = make_graph(whole)
    scaffold_data, scaffold = make_graph(scaffold)
    if whole_data == None or scaffold_data == None:
        return None, None
    
    scaffold_data.indices = list(whole.GetSubstructMatches(scaffold)[0])
    return whole_data, scaffold_data


def generate_order(whole, scaff):
    # 0 = add atom
    # 1 = add edge
    # 2 = terminate graph
    
    matches = list(whole.GetSubstructMatch(scaff))

    atom_ids = [x for x in range(len(whole.GetAtoms())) if x not in matches]

    direct = []
    atom_order = []
    edge_order = []
    action_order = []
    paths = 1

    for idx in atom_ids:
        common = set([x.GetIdx() for x in whole.GetAtomWithIdx(idx).GetNeighbors()]) & set(matches)
        if len(common) != 0:
            direct.append([idx, list(common)[0]])

    direct = shuffle(direct)

    for start_node in direct:
        current_node, tar = start_node

        atom_order.append([current_node, tar])
        edge_order.append([])
        action_order.append(0)

        terminate = False
        while terminate is False:
            last = current_node
            source_atoms = [x[0] for x in atom_order]
            remain_atoms = [x for x in atom_ids if x not in source_atoms]

            if len(remain_atoms) == 0:
                terminate = True

            neighbors = whole.GetAtomWithIdx(last).GetNeighbors()

            links = [x.GetIdx() for x in neighbors if x.GetIdx() in remain_atoms]

            if len(links) == 0:
                if source_atoms.index(current_node) == 0:
                    terminate=True
                current_node = source_atoms[source_atoms.index(current_node) - 1]
                continue

            if len(links) == 1:
                current_node = links[0]
            else:
                current_node = random.sample(links, 1)[0]
                paths += 1

            action_order.append(0)

            edge_neighbors = whole.GetAtomWithIdx(current_node).GetNeighbors()

            edge_list = [x.GetIdx() for x in edge_neighbors]
            edge_list = [x for x in edge_list if x not in remain_atoms and x != last]

            for edge in edge_list:
                action_order.append(1)

            atom_order.append([current_node, last])
            edge_order.append(edge_list)

    action_order.append(2)
                
    return atom_order, edge_order, action_order, paths


def generate_graphs(whole, scaff):
    graph_order = []
    action_order = []
    atom_types = []
    append_bond_types = []
    connect_bond_types = []
    total_latest = []
    total_scaffold_mask = []
    lengths = []
    
    atom_order, edge_order, graph_actions, paths = generate_order(Chem.MolFromSmiles(whole), Chem.MolFromSmiles(scaff))
    
    whole_graph, current = make_analog_pair(whole, scaff)
    
    if whole_graph == None:
        return None, None, None, None, None, None
    
    latest = [0] * len(current.x)
    scaffold_mask = [1] * len(current.x)
    
    total_latest.extend(latest)
    total_scaffold_mask.extend(scaffold_mask)
    
    lengths.append(len(current.x))
    
    graph_order.append(copy(current))

    for r in range(len(atom_order)):
        target_atom = atom_order[r][1]
        source_atom = atom_order[r][0]
        edge_list = edge_order[r]

        scaff_i = len(current.x)
        scaff_j = current.indices.index(target_atom)
        
        indices = list(set((whole_graph.edge_index == source_atom).nonzero()[:, 1].tolist()) & set((whole_graph.edge_index == target_atom).nonzero()[:, 1].tolist()))
        
        atom_type = whole_graph.x[source_atom]
        edge_type = whole_graph.edge_features[indices[0]]
        
        atom_labels = [0] * len(current.x)
        atom_labels[scaff_j] = atom_type
        
        bond_labels = [0] * len(current.x)
        bond_labels[scaff_j] = edge_type
        
        action_labels = [-1] * len(current.x)
        action_labels[scaff_j] = 0
        
        atom_types.extend(atom_labels)
        append_bond_types.extend(bond_labels)
        connect_bond_types.extend([0] * len(current.x))
        action_order.extend(action_labels)
        scaffold_mask = scaffold_mask + [0]
        total_scaffold_mask.extend(scaffold_mask)
        
        new_latest = torch.Tensor(latest + [1])
        total_latest.extend(new_latest)
                
        new_atom = torch.Tensor([whole_graph.x[source_atom]])
        new_edge_index = torch.Tensor([[scaff_i, scaff_j], [scaff_j, scaff_i]])
        new_edge_features = whole_graph.edge_features[indices]
        
        current.x = torch.cat([current.x, torch.zeros(1)])
        current.indices.append(source_atom)
        
        lengths.append(len(current.x))
        
        addition = Data(x=new_atom, edge_index=new_edge_index, edge_features=new_edge_features)
        
        graph_order.append(addition)
                
        if len(edge_list) == 0:
            latest.append(0)
            continue
        else:
            for edge in edge_list:
                scaff_j = current.indices.index(edge)
                new_edge_index = torch.Tensor([[scaff_i, scaff_j], [scaff_j, scaff_i]])

                indices = list(set((whole_graph.edge_index == source_atom).nonzero()[:, 1].tolist()) & set((whole_graph.edge_index == edge).nonzero()[:, 1].tolist()))
                new_edge_features = whole_graph.edge_features[indices]
                
                edge_type = whole_graph.edge_features[indices[0]]
                
                bond_labels = [0] * len(current.x)
                bond_labels[scaff_j] = edge_type
                
                action_labels = [-1] * len(current.x)
                action_labels[scaff_j] = 1
                
                total_scaffold_mask.extend(scaffold_mask)

                new_latest = torch.Tensor(latest + [2])
                total_latest.extend(new_latest)
                
                atom_types.extend([0] * len(current.x))
                append_bond_types.extend([0] * len(current.x))
                connect_bond_types.extend(bond_labels)
                action_order.extend(action_labels)
                
                lengths.append(len(current.x))
                
                addition = Data(x=torch.Tensor([]), edge_index=new_edge_index, edge_features=new_edge_features)

                graph_order.append(addition)
                                
            latest.append(0)
                
    atom_types.extend([0] * len(current.x))
    append_bond_types.extend([0] * len(current.x))
    connect_bond_types.extend([0] * len(current.x))
    
    action_labels = [-1] * len(current.x)
    action_labels[-1] = 2
    
    action_order.extend(action_labels)
    del graph_order[0].indices
                    
    return graph_order, (torch.Tensor(action_order), graph_actions), (torch.Tensor(atom_types), torch.Tensor(append_bond_types), 
                                                     torch.Tensor(connect_bond_types)), (torch.Tensor(total_latest), torch.Tensor(total_scaffold_mask),
                                                                                                torch.Tensor(lengths)), whole_graph, paths


def to_long(arr):
    return arr.long()


def samp_to_long(item):
    # for item in graphs:
    for step in item[0]:
        step.apply(to_long)
    item[1].apply(to_long, 'x', 'edge_index', 'edge_features', 'node_actions', 'atom_types', 'append_bond_types', 'connect_bond_types', 'latest', 'scaffold', 'history_batch', 'seq_lengths')

    
def combine_data(data1, data2):
    return Data(x=torch.cat([data1.x, data2.x]), edge_index=torch.cat([data1.edge_index, data2.edge_index], dim=1), 
                edge_features=torch.cat([data1.edge_features, data2.edge_features]))

class Collater:
    def __init__(self):
        self.original_collater = torch_geometric.loader.dataloader.Collater(None, None)

    def __call__(self, batch):
        # k_length = np.cumsum([len(x) for x in batch])
        # batch = list(itertools.chain.from_iterable(batch))
                
        batch, wholes = list(zip(*batch))
        
        batches = []
        length = 0
        
        for steps in batch:
            scaff = steps[0]
            graphs = [scaff]
                        
            for step in steps[1:]:
                scaff = combine_data(scaff, step)
                graphs.append(scaff)
                                
            batches.append(Batch.from_data_list(graphs))
            
        batch = list(zip(batches, wholes))
        elem = batch[0]
        
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.original_collater(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')
        

class GraphLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(), num_workers=num_workers)
    
    
def graph_to_mol(graph, sanitize=True, scaffold=True):
    node_features = graph.x
    edge_index = graph.edge_index[:, ::2]
    edge_features = graph.edge_features[::2]
    if scaffold:
        is_scaffold = graph.scaffold
        scaffold_atoms = torch.arange(len(node_features))[is_scaffold.long()].tolist()
    
    mol = Chem.RWMol()
    adjust_hs = []
    Ns = []
    aromatic_atoms = []
    atom_symbols = [atom_list[int(x.item())] for x in node_features]

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(atom_symbols)):
        symbol, charge, Hs = atom_symbols[i]
        a = Chem.Atom(symbol)
        a.SetFormalCharge(int(charge))
        a.SetNumExplicitHs(int(Hs))
        if symbol == 'N':
            Ns.append(i)
        
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    bond_types = [bond_list[int(x.item())] for x in edge_features]
    for i in range(len(edge_features)):
        first, second = edge_index[0][i], edge_index[1][i]
        ifirst = node_to_idx[first.item()]
        isecond = node_to_idx[second.item()]
        
        if scaffold:
        
            if (is_scaffold[ifirst] == 1 and is_scaffold[isecond] != 1):
                adjust_hs.append(ifirst)

            if (is_scaffold[ifirst] != 1 and is_scaffold[isecond] == 1):
                adjust_hs.append(isecond)
        
        bond_type = bond_types[i]
        
        if bond_type == Chem.rdchem.BondType.AROMATIC:
            aromatic_atoms.append(ifirst)
            aromatic_atoms.append(isecond)
        
        mol.AddBond(ifirst, isecond, bond_type)
        
    if scaffold:
        for idx in adjust_hs:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            elif idx in Ns and idx in aromatic_atoms:
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        
    if sanitize:
        Chem.SanitizeMol(mol)

    # Convert RWMol to Mol object
    mol = mol.GetMol()   
    return mol


core_list = []
du = Chem.MolFromSmiles('*')
Hs = Chem.MolFromSmiles('[H]')

def get_cores(smiles):
    new_mols = []
    
    # try:
    #     bm_smiles = Chem.CanonSmiles(MurckoScaffoldSmiles(smiles))
    # except:
    #     return (r, None)
    # if bm_smiles != smiles:
    #     new_mols.append(bm_smiles)
    
    bm_smiles = smiles
    
    mol = Chem.MolFromSmiles(bm_smiles)
    # mol = Chem.MolFromSmiles(smiles)
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy >= 50:
        return None
    
    try:
        hierarch = RecapDecompose(mol)
    except:
        return None
    
    cores = list(hierarch.GetAllChildren().values())
    cores = [x.mol for x in cores]
    cores = [x for x in cores if x.GetNumHeavyAtoms() >= num_heavy * 0.5]
    
    cores = [Chem.ReplaceSubstructs(x, du, Hs, replaceAll=True)[0] for x in cores]
    new_mols = new_mols + [Chem.CanonSmiles(Chem.MolToSmiles(x)) for x in cores]
    new_mols = [x for x in new_mols if new_mols != None]
    # if len(new_mols) == 0:
    #     zeros += 1
    
    return new_mols
