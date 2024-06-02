import json
import codecs
import random
import numpy as np
import networkx as nx
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from typing import List, Union

# --- Class variables
ATOM_FEATURES = {
    'atomic_num': [6.0, 8.0],
    'degree': [1, 2, 3, 4],
    'num_Hs': [0.0, 1.0, 2.0],
    'num_Os': [0.0, 1.0, 2.0],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
}

# --- Total number of atom features
ATOM_FDIM = sum(len(choices) for choices in ATOM_FEATURES.values()) + 1
# --- Number of bond features
BOND_FDIM = 14

def mol_to_nx(mol, spec, atom_rep=True):
    '''
    text
    '''

    # --- Create graph object
    G = nx.Graph()
        
    # --- For each atom in molecule
    for atom in mol.GetAtoms():
        # --- Add a node to graph and create one-hot encoding vector for atom features
        map_num = atom.GetAtomMapNum()
        G.add_node(map_num, x=get_atom_features(atom))
            
    # --- For each bond in molecule
    for bond in mol.GetBonds():
        # ---
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_map = begin.GetAtomMapNum()
        end_map = end.GetAtomMapNum()
        # --- Add edge to graph and create one-hot encoding vector of bond features
        G.add_edge(begin_map, end_map, edge_attr=get_bond_features(bond))

    if atom_rep == False:
        # --- Normalize spectra to 1.0
        max_intensity = np.max(spec)
        norm_spec = 1.0 * (spec / max_intensity)
        # --- Set spectra to graph
        G.graph['spectrum'] = torch.FloatTensor(norm_spec)
        
    return G
    
def get_atom_features(atom) -> List[Union[bool, int, float]]:
    '''
    Builds a feature vector for an atom

    :param atom: An RDKit atom
    :return: A list containing the atom features
    '''
    # # --- For a shorter feature vector
    # num_Os = 0
    # if atom is None:
    #     features = [0] * ATOM_FDIM
    # else:
    #     for a in atom.GetNeighbors():
    #         if a.GetAtomicNum() == 8:
    #             num_Os += 1.0

    #     if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
    #         hybrid = 2.0
    #     elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
    #         hybrid = 3.0

    #     if atom.GetIsAromatic() == True:
    #         aroma = 1.0
    #     elif atom.GetIsAromatic() == False:
    #         aroma = 0.0

    #     features = atom.GetAtomicNum(), atom.GetDegree(), atom.GetTotalNumHs(), hybrid, aroma

    # --- For one-hot encoding featue vector
    num_Os = 0
    if atom is None:
        features = [0] * ATOM_FDIM
    else:
        for a in atom.GetNeighbors():
            if a.GetAtomicNum() == 8:
                num_Os += 1.0
        # --- Get the values of all the atom features and add all up to the feature vector
        features = one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree']) + \
            one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
            one_hot_encoding(num_Os, ATOM_FEATURES['num_Os']) + \
            one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0]

    return features        

def get_bond_features(bond) -> List[Union[bool, int, float]]:
    '''
    Builds a features vector for a bond

    :params bond: An RDKit bond
    :return: A list containing the bond features
    '''

    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        # Get the bond type and create one-hot enconding vector
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            typ = 1.0
        elif bt == Chem.rdchem.BondType.DOUBLE:
            typ = 2.0
        elif bt == Chem.rdchem.BondType.AROMATIC:
            typ = 3.0

        if bond.GetIsConjugated() == True:
            conj = 1.0
        else:
            conj = 0.0

        if bond.IsInRing() == True:
            ring = 1.0
        else:
            ring = 0.0
        fbond = [ typ, conj, ring]

    return fbond

def one_hot_encoding(value:int, choices:List[int]) -> List[int]:
    '''
    Creates a one-hot encoding
    '''

    # --- Create a zero atom feature vector
    encoding = [0.0] * len(choices)
    # --- Find the index value of
    index = choices.index(value)
    # --- Set value to 1
    encoding[index] = 1.0

    return encoding

def count_atoms(mol, atomic_num):
    '''
    text
    '''

    # --- 
    num_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atomic_num:
            num_atoms += 1
    return num_atoms


class XASDataset_mol(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property
    def raw_file_names(self):
        return ['data_coronene_new.json']
        #return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['data_mol_long.pt']
    
    def process(self):
        '''
        Text
        '''

        # --- List to store data
        data_list = []

        # --- Load the raw data from json file
        dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        dictionaires = json.load(dat)

        # --- Create list with all the molecule names
        all_names = list(dictionaires[0].keys())
        print(f'Total number of molecules {len(all_names)}')

        # --- 
        idx = 0
        
        for name in all_names:
            # --- 
            smiles = dictionaires[0][name][0]
            mol = Chem.MolFromSmiles(smiles)
            # ---
            atom_spec = dictionaires[1][name]

            # --- Create arrays of dataset
            pos = dictionaires[0][name][1]
            positions = np.array(pos)
            z_num = dictionaires[0][name][2]
            z = np.array(z_num)
            atom_count = count_atoms(mol, 6)

            tot_spec = np.zeros(len(atom_spec[str(0)]))

            for j in range(atom_count):
                # --- Sum up all atomic spectra
                tot_spec += atom_spec[str(j)]

            # --- Create graph object
            gx = mol_to_nx(mol, tot_spec, atom_rep=False)
            # --- Convert to pyg
            pyg_graph = from_networkx(gx)
            pyg_graph.pos = torch.from_numpy(positions)
            pyg_graph.z = torch.from_numpy(z)
            pyg_graph.idx = idx
            pyg_graph.smiles = smiles
            data_list.append(pyg_graph)
            idx += 1

        random.Random(258).shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    

class XASDataset_atom(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property
    def raw_file_names(self):
        return ['data_coronene_new.json']
        #return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['data_atom_long_317.pt']
    
    def process(self):
        '''
        Text
        '''

        # --- List to store data
        data_list = []

        # --- Load the raw data from json file
        dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        dictionaires = json.load(dat)

        # --- Create list with all the molecule names
        all_names = list(dictionaires[0].keys())
        print(f'Total number of molecules {len(all_names)}')

        # --- 
        idx = 0
        individual_graphs = False

        for name in all_names:
            # --- 
            smiles = dictionaires[0][name][0]
            mol = Chem.MolFromSmiles(smiles)
            # ---
            atom_spec = dictionaires[1][name]

            # --- Create arrays of dataset
            pos = dictionaires[0][name][1]
            positions = np.array(pos)
            z_num = dictionaires[0][name][2]
            z = np.array(z_num)
            atom_count = 0
            for atom in mol.GetAtoms():
                atom_count += 1

            if individual_graphs == False:
                # --- For all atom spectra in feature vector
                spec_list = torch.zeros([atom_count, 200])
        
                for x in range(len(atom_spec)):
                    sp = atom_spec[str(x)]
                    spec_list[x] = torch.DoubleTensor(sp)

                map_dict = {}
                for atom in enumerate(mol.GetAtoms()):
                    map = atom[1].GetAtomMapNum()
                    map_dict[atom[0]] = map

                # --- Create graph object
                gx = mol_to_nx(mol, spec_list, atom_rep=True)

                # --- Convert to pyg
                pyg_graph = from_networkx(gx)
                pyg_graph.spectrum = spec_list
                pyg_graph.pos = torch.from_numpy(positions)
                pyg_graph.z = torch.from_numpy(z)
                pyg_graph.idx = idx
                pyg_graph.smiles = smiles
                data_list.append(pyg_graph)
                idx += 1
            else:
                for x, atom in zip(range(len(atom_spec)), mol.GetAtoms()):
                    spectrum = atom_spec[str(x)]

                    gx = mol_to_nx(mol, spectrum, atom_rep=True)

                    pyg_graph = from_networkx(gx)
                    pyg_graph.spectrum = torch.FloatTensor(spectrum)
                    pyg_graph.pos = torch.from_numpy(positions)
                    pyg_graph.z = torch.from_numpy(z)
                    pyg_graph.idx = idx
                    pyg_graph.atom_num = atom.GetAtomMapNum()
                    pyg_graph.smiles = smiles
                    data_list.append(pyg_graph)
                    idx += 1


        random.Random(258).shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])