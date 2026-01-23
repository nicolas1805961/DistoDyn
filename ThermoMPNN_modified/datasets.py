import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from protein_mpnn_utils import alt_parse_PDB, parse_PDB, parse_CIF
from cache import cache
from glob import glob
import math


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'


@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    return parse_PDB(pdb_file)

def wt_sort_key(f):
    basename = os.path.basename(f)
    wt_name = basename.split("_model")[0].replace("distogram_", "")
    # assume wt_name is like 'protein_32'
    num = int(wt_name.split("_")[-1])
    return num


@dataclass
class Mutation:
    position: int
    wildtype: str
    mutation: str
    ddG: Optional[float] = None
    #pdb: Optional[str] = ''


def seq1_index_to_seq2_index(align, index):
    """Do quick conversion of index after alignment"""
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != '-':
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break
    
    # now the index in seq 2 cooresponding to aligned index
    if align.seqB[aln_idx] == '-':
        return None

    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == '-':
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx


class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve
        self.use_distogram_features = cfg.training.use_distogram_features
        self.mode = cfg.training.mode
        self.disto_location = cfg.data_loc.megascale_distograms

        data = torch.load('cutoff_matrix.pt', map_location="cpu")
        self.cutoff_mean = data["cutoff_mean"]   # [20, 20]
        self.cutoff_sigma = data["cutoff_sigma"] # [20, 20]
        self.aa_to_idx = data["aa_to_idx"]
        self.aa_list = data["aa_list"]

        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        #df = pd.read_csv(fname, usecols=["dG_ML", "mut_type", "WT_name", "aa_seq", "wt_seq"])
        df = pd.read_csv(fname, usecols=["WT_name", "name", "ddG_ML", "mut_type", "new_name", "aa_seq", "wt_seq", "split"])
        df = df[df["split"] == split].copy()

        mask = df["WT_name"].str.contains(r"\.pdb_.+$", regex=True, na=False)
        df = df[~mask]

        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        #with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
        #    splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [], 
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        if 'reduce' not in cfg:
            cfg.reduce = ''

        self.wt_seqs = {}
        self.mut_rows = {}

        self.wt_names = sorted({name.rsplit("_", 1)[0] for name in df["new_name"].values},key=lambda x: int(x.split("_")[1]))

        #print(self.wt_names)
        
        #import re
#
        #pattern = '|'.join(map(re.escape, self.wt_names))
        #result = set(
        #            df.loc[
        #                df['new_name'].str.contains(pattern, na=False) &
        #                df['WT_name'].str.contains(r'\.pdb_', na=False),
        #                'WT_name'
        #            ]
        #            #.str.split('.', n=1)
        #            #.str[0]
        #            .tolist()
        #        )
        #print(len(result))
        #print(result)


        print(len(self.wt_names))

        if self.use_distogram_features or self.mode != 'vanilla':
            print("Loading distogram files...")

            #self.distogram_files = glob(os.path.join('/pasteur/appa/scratch/nportal/boltz/stability_prediction/results_wt', '**', '**', "distogram*.pkl"), recursive=True)
            # Get all distogram files (unique)
            all_files = list(set(glob(os.path.join(self.disto_location, '**', '**', "distogram*.pt"), recursive=True)))

            all_files_bin_edge = list(set(glob(os.path.join(self.disto_location, '**', '**', "bin_edge*.pt"), recursive=True)))
            
            # Keep only files whose name matches a WT in this split
            # Keep only files whose basename matches exactly a WT in this split
            self.distogram_files = [
                f for f in all_files
                if os.path.basename(f).startswith("distogram_") and
                os.path.basename(f).split("_model")[0].replace("distogram_", "") in self.wt_names
            ]

            self.bin_edge_files = [
                f for f in all_files_bin_edge
                if os.path.basename(f).startswith("bin_edge_") and
                os.path.basename(f).split("_model")[0].replace("bin_edge_", "") in self.wt_names
            ]

            self.distogram_files = sorted(self.distogram_files, key=wt_sort_key)

            self.bin_edge_files = sorted(self.bin_edge_files, key=wt_sort_key)

            print(f"Number of distogram files for split '{split}':", len(self.distogram_files))
        


        #if self.split == 'all':
        #    all_names = splits['train'] + splits['val'] + splits['test']
        #    self.split_wt_names[self.split] = all_names
        #else:
        #    if cfg.reduce == 'prot' and self.split == 'train':
        #        n_prots_reduced = 58
        #        self.split_wt_names[self.split] = np.random.choice(splits["train"], n_prots_reduced)
        #    else:
        #        self.split_wt_names[self.split] = splits[self.split]
#
        #self.wt_names = self.split_wt_names[self.split]

        for wt_name in tqdm(self.wt_names):
            self.mut_rows[wt_name] = (
                    df.query('new_name.str.contains(@wt_name + "_") and mut_type != "wt"', engine="python")
                    .reset_index(drop=True)
                )
            self.wt_seqs[wt_name] = df.loc[df["new_name"].str.contains(wt_name + '_', regex=False), "wt_seq"].unique()
            assert len(self.wt_seqs[wt_name]) == 1, f"Multiple wt_seqs found for {wt_name}"
            self.wt_seqs[wt_name] = self.wt_seqs[wt_name][0]

            #self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            #try:
            #    wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            #    if type(cfg.reduce) is float and self.split == 'train':
            #        self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(cfg.reduce), replace=False)
#
            #    self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
            #except Exception as e:
            #    wt_rows = df.query('WT_name == @wt_name').reset_index(drop=True)
            #    if type(cfg.reduce) is float and self.split == 'train':
            #        self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(cfg.reduce), replace=False)
#
            #    self.wt_seqs[wt_name] = wt_rows.wt_seq[0]
    
    def seq_to_aa_idx(self, seq):
        one_to_three = {
                            "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
                            "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
                            "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
                            "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"
                        }
        return torch.tensor([self.aa_to_idx[one_to_three[aa]] for aa in seq], dtype=torch.long)


    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        mut_data = self.mut_rows[wt_name]
        wt_seq = self.wt_seqs[wt_name]

        if self.use_distogram_features or self.mode != 'vanilla':
            disto_file = self.distogram_files[index]
            bin_edge_file = self.bin_edge_files[index]
            assert wt_name == os.path.basename(disto_file).split("_model")[0].replace("distogram_", ""), f"{wt_name} vs {disto_file}"
            disto = torch.load(disto_file, map_location="cuda")
            disto = disto.detach()
            if self.mode != 'vanilla':
                wt_idx = self.seq_to_aa_idx(wt_seq)
                cutoff_mean_ij = self.cutoff_mean[wt_idx[:, None], wt_idx[None, :]]  # [L,L]
                cutoff_sigma_ij = self.cutoff_sigma[wt_idx[:, None], wt_idx[None, :]]  # [L,L]
                cutoff = cutoff_mean_ij + 1.645 * cutoff_sigma_ij

                bin_edges = torch.load(bin_edge_file, map_location="cpu")[0]

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # shape = 61
                # Add first and last bins to make 64 centers
                first_center = bin_edges[0] - (bin_edges[1] - bin_edges[0])/2
                last_center = bin_edges[-1] + (bin_edges[-1] - bin_edges[-2])/2
                bin_centers = torch.cat([first_center[None], bin_centers, last_center[None]])  # shape = 64
                bin_centers = bin_centers.cuda()
                cutoff = cutoff.cuda()
                bins_to_sum = bin_centers[None, None, :] <= cutoff[:, :, None]  # [L,L,64], boolean
            else:
                bins_to_sum = None
        else:
            disto = None
            bins_to_sum = None

        # Open and load
        #with open(disto_file, "rb") as f:
        #    data = pickle.load(f)
        #    disto = torch.from_numpy(data['distogram']['logits']).unsqueeze(0)  # add batch dim
        #    disto = self.normalize_distogram(disto)

        #print(f"Loading {wt_name}...")

        wt_name = wt_name.split(".pdb")[0].replace("|",":")

        #pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
        pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.cif")
        #pdb = parse_pdb_cached(self.cfg, pdb_file)
        pdb = parse_CIF(pdb_file)
        assert len(pdb[0]["seq"]) == len(wt_seq)
        pdb[0]["seq"] = wt_seq

        mutations = []
        for i, row in mut_data.iterrows():
            # no insertions, deletions, or double mutants
            if "ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type:
                continue
            assert len(row.aa_seq) == len(wt_seq), f"{len(row.aa_seq)} vs {len(wt_seq)}, {row.aa_seq} vs {wt_seq}"
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1
            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut

            if row.ddG_ML == '-':
                continue # filter out any unreliable data

            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)
            mutations.append(Mutation(idx, wt, mut, ddG))
            #mutations.append(Mutation(idx, wt, mut, ddG, wt_name))

        return pdb, mutations, disto, bins_to_sum



class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        filename = self.cfg.data_loc.fireprot_csv

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.fireprot_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "homologue-free": [],
            "all": []
        }

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query('pdb_id_corrected == @wt_name').reset_index(drop=True)
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]


    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[seq]

        pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{data.pdb_id_corrected[0]}.pdb")
        pdb = parse_pdb_cached(self.cfg, pdb_file)

        mutations = []
        for i, row in data.iterrows():
            try:
                pdb_idx = row.pdb_position
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
                
            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(seq, pdb[0]['seq'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]

            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            mut = Mutation(pdb_idx, pdb[0]['seq'][pdb_idx], row.mutation, ddG, wt_name)
            mutations.append(mut)

        return pdb, mutations


class ddgBenchDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname):

        self.cfg = cfg
        self.pdb_dir = pdb_dir

        df = pd.read_csv(csv_fname)
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()

        for wt_name in self.wt_names:
            wt_name_query = wt_name
            wt_name = wt_name[:-1]
            self.mut_rows[wt_name] = df.query('PDB == @wt_name_query').reset_index(drop=True)
            if 'S669' not in self.pdb_dir:
                self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = [wt_name[-1]]

        wt_name = wt_name.split(".pdb")[0][:-1]
        mut_data = self.mut_rows[wt_name]

        pdb_file = os.path.join(self.pdb_dir, wt_name + '.pdb')

        # modified PDB parser returns list of residue IDs so we can align them easier
        pdb = alt_parse_PDB(pdb_file, chain)
        resn_list = pdb[0]["resn_list"]

        mutations = []
        for i, row in mut_data.iterrows():
            mut_info = row.MUT
            wtAA, mutAA = mut_info[0], mut_info[-1]
            try:
                pos = mut_info[1:-1]
                pdb_idx = resn_list.index(pos)
            except ValueError:  # skip positions with insertion codes for now - hard to parse
                continue
            try:
                assert pdb[0]['seq'][pdb_idx] == wtAA
            except AssertionError:  # contingency for mis-alignments
                # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
                if 'S669' in self.pdb_dir:
                    gaps = [g for g in pdb[0]['seq'] if g == '-']
                else:
                    gaps = [g for g in pdb[0]['seq'][:pdb_idx + 10] if g == '-']                

                if len(gaps) > 0:
                    pdb_idx += len(gaps)
                else:
                    pdb_idx += 1

                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == wtAA
            ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor([row.DDG * -1.], dtype=torch.float32)
            mut = Mutation(pdb_idx, pdb[0]['seq'][pdb_idx], mutAA, ddG, wt_name)
            mutations.append(mut)

        return pdb, mutations


class ComboDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        datasets = []
        if "fireprot" in cfg.datasets:
            fireprot = FireProtDataset(cfg, split)
            datasets.append(fireprot)
        if "megascale" in cfg.datasets:
            mega_scale = MegaScaleDataset(cfg, split)
            datasets.append(mega_scale)
        self.mut_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.mut_dataset)

    def __getitem__(self, index):
        return self.mut_dataset[index]


