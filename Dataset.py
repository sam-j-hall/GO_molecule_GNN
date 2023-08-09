import pandas as pd
import os 
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolfiles import  MolFromXYZFile
from rdkit.Chem import rdDetermineBonds
import rdkit
import networkx as nx
import torch
import re
import pickle
import torch_geometric
from torch_geometric.data import Data
from typing import List, Tuple, Union
from torch_geometric.data import InMemoryDataset,InMemoryDataset,download_url
from torch_geometric.utils.convert import from_networkx
import random
import json


class XASDataset(InMemoryDataset):
    
    ATOM_FEATURES = {
            'atomic_num': [6,8],
            'degree': [0, 1, 2, 3, 4],
           # 'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3
                ],
            }

    ATOM_FDIM=sum(len(choices)  for choices in ATOM_FEATURES.values()) + 1
    BOND_FDIM = 14

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
            
 
    @property
    def raw_file_names(self):
        #return ['input.pkl']
        return ['data.json']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def onek_encoding_unk(self,value: int, choices: List[int]) -> List[int]:
        """
        Creates a one-hot encoding with an extra category for uncommon values.

        :param value: The value for which the encoding should be one.
        :param choices: A list of possible values.
        :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
                 If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
        """
        encoding = [0] * (len(choices))
        index = choices.index(value) #if value in choices 
        encoding[index] = 1

        return encoding

    def mol_to_nx(self,mol,spec,atom_id):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                      x=self.atom_features(atom))
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       edge_attrs=self.bond_features(bond))
        
        spectrum=np.array(spec)
        max_intensity = np.max(spectrum)
        norm_spec=10.0*(spectrum/max_intensity)
        G.graph['y']=norm_spec
        G.graph['index']=atom_id
        return G

    def atom_features(self,atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for an atom.

        :param atom: An RDKit atom.
        :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
        :return: A list containing the atom features.
        """
        if atom is None:
            features = [0] * ATOM_FDIM
        else:
            features = self.onek_encoding_unk(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']) + \
                self.onek_encoding_unk(atom.GetTotalDegree(), self.ATOM_FEATURES['degree']) + \
                self.onek_encoding_unk(int(atom.GetChiralTag()), self.ATOM_FEATURES['chiral_tag']) + \
                self.onek_encoding_unk(int(atom.GetTotalNumHs()), self.ATOM_FEATURES['num_Hs']) + \
                self.onek_encoding_unk(int(atom.GetHybridization()), self.ATOM_FEATURES['hybridization']) + \
                [1 if atom.GetIsAromatic() else 0] 
             #   [atom.GetMass() * 0.1]  # scaled to about the same range as other features
            if functional_groups is not None:
                features += functional_groups
        return features
      #   self.onek_encoding_unk(atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge']) + \
    def bond_features(self,bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for a bond.

        :param bond: An RDKit bond.
        :return: A list containing the bond features.
        """
        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                int(bt == Chem.rdchem.BondType.SINGLE),
                int(bt == Chem.rdchem.BondType.DOUBLE),
                int(bt == Chem.rdchem.BondType.TRIPLE),
                int(bt == Chem.rdchem.BondType.AROMATIC),
                int(bond.GetIsConjugated() if bt is not None else 0),
                int(bond.IsInRing() if bt is not None else 0)
            ]
            fbond += self.onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond

    def count_atoms(self,mol,atomic_num):
        #Works for all atoms, input atomic_num
        num_oxygens = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == atomic_num:  # Atomic number 8 corresponds to oxygen
                num_oxygens += 1
        return num_oxygens

        


    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]
        
        
        with open(self.raw_paths[0], "rb") as file:
                #dictionaries = pickle.load(file)
                dictionaries=json.load(file)
        
        tot_ids=list(dictionaries[0].keys())
        atom_count=[]
        for mol_id in tot_ids :
            mol=Chem.MolFromSmiles(dictionaries[0][mol_id])
            tot_atoms=self.count_atoms(mol,6)
            atom_count.append(tot_atoms)

        print(atom_count)    
        sum_atoms=sum(atom_count)
        
        idx=0
        data_list=[]
        for i in range(len(tot_ids)):
            #print(tot_ids[i])

            test_mol=Chem.MolFromSmiles(dictionaries[0][tot_ids[i]])

#             for atom in test_mol.GetAtoms():
#                 feat=self.atom_features(atom)

#             for bond in test_mol.GetBonds():
#                 bond_feat=self.bond_features(bond)

            test_spec=dictionaries[1][tot_ids[i]]

            for j in range(int(atom_count[i])):
                spec_dict=test_spec[str(j)]
                gx=self.mol_to_nx(test_mol,spec_dict,j)
                pyg_graph = from_networkx(gx)
                pyg_graph.idx=idx
                data_list.append(pyg_graph)
                #print('Element added:',idx)
                idx+=1
        
        
        random.shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
