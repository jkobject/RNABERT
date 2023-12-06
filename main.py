import torch
from .utils import (
    get_config,
    BertModel,
    BertForMaskedLM,
)
from .dataload import DATA
import numpy as np
import os
from Bio import SeqIO
import random
import pandas as pd
import copy
import itertools

FILE_LOC = os.path.dirname(os.path.realpath(__file__))
# https://github.com/mana438/RNABERT
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class RNABERT:
    def __init__(
        self,
        batch_size=20,
        config=FILE_LOC + "/RNA_bert_config.json",
        pretrained_model=FILE_LOC + "/bert_mul_2.pth",
        device="cuda",
    ):
        self.file_location = os.path.dirname(os.path.realpath(__file__))
        self.batch_size = batch_size
        self.config = get_config(file_path=config)
        self.max_length = self.config.max_position_embeddings
        self.maskrate = 0
        self.mag = 1

        self.config.hidden_size = self.config.num_attention_heads * self.config.multiple
        model = BertModel(self.config)
        self.model = BertForMaskedLM(self.config, model)
        self.model.to(device)
        print("device: ", device)
        self.device = device
        if device == "cuda":
            self.model = torch.nn.DataParallel(self.model)  # make parallel
        self.model.load_state_dict(torch.load(pretrained_model))

    def __call__(self, fasta_file):
        print("-----start-------")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # names = []
        # seqs = []
        # for record in SeqIO.parse(fasta_file, "fasta"):
        #    gapped_seq = str(record.seq).upper()
        #    gapped_seq = gapped_seq.replace("T", "U")
        #    seq = gapped_seq.replace("-", "")
        #    if (
        #        set(seq) <= set(["A", "T", "G", "C", "U"])
        #        and len(list(seq)) < self.max_length
        #    ):
        #        seqs.append(seq)
        #        names.append(record.id)
        # family = np.tile(np.zeros(len(seqs)), self.mag)
        # seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)
        # k = 1
        # kmer_seqs = kmer(seqs, k)
        # masked_seq, low_seq = mask(kmer_seqs, rate=0, mag=self.mag)
        # kmer_dict = make_dict(k)
        # masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))

        # dataloader = torch.utils.data.DataLoader(
        #    MyDataset("SHOW", low_seq, masked_seq, family, seqs_len),
        #    self.batch_size,
        #    shuffle=False,
        # )
        data = DATA(config=self.config, batch=self.batch_size)
        seqs, names, dataloader = data.load_data_EMB([fasta_file])
        features = self.make_feature(self.model, dataloader, seqs)
        features = np.array([np.array(embedding).sum(0) for embedding in features])

        return pd.DataFrame(features, index=names)

    def make_feature(self, model, dataloader, seqs):
        model.eval()
        torch.backends.cudnn.benchmark = True
        encoding = []
        for batch in dataloader:
            data, label, seq_len = batch
            inputs = data.to(self.device)
            _, _, encoded_layers = model(inputs)
            encoding.append(encoded_layers.cpu().detach().numpy())
        encoding = np.concatenate(encoding, 0)

        embedding = []
        for e, seq in zip(encoding, seqs):
            embedding.append(e[: len(seq)].tolist())

        return embedding


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_type,
        low_seq,
        masked_seq,
        family,
        seq_len,
        low_seq_1=None,
        masked_seq_1=None,
        family_1=None,
        seq_len_1=None,
        common_index=None,
        common_index_1=None,
        SS=None,
        SS_1=None,
    ):
        self.train_type = train_type
        self.data_num = len(low_seq)
        self.low_seq = low_seq
        self.low_seq_1 = low_seq_1
        self.masked_seq = masked_seq
        self.masked_seq_1 = masked_seq_1
        self.family = family
        self.family_1 = family_1
        self.seq_len = seq_len
        self.seq_len_1 = seq_len_1
        self.common_index = common_index
        self.common_index_1 = common_index_1
        self.SS = SS
        self.SS_1 = SS_1

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_low_seq = self.low_seq[idx]
        out_masked_seq = self.masked_seq[idx]
        out_family = self.family[idx]
        out_seq_len = self.seq_len[idx]
        if (
            self.train_type == "MLM"
            or self.train_type == "MUL"
            or self.train_type == "SSL"
        ):
            out_low_seq_1 = self.low_seq_1[idx]
            out_masked_seq_1 = self.masked_seq_1[idx]
            out_family_1 = self.family_1[idx]
            out_seq_len_1 = self.seq_len_1[idx]

        if self.train_type == "MUL" or self.train_type == "SSL":
            out_common_index = self.common_index[idx]
            out_common_index_1 = self.common_index_1[idx]

        if self.train_type == "SSL":
            out_SS = self.SS
            out_SS_1 = self.SS_1

        # if self.train_type == "SHOW":
        #     out_SS = self.SS

        if self.train_type == "MLM":
            return (
                out_low_seq,
                out_masked_seq,
                out_family,
                out_seq_len,
                out_low_seq_1,
                out_masked_seq_1,
                out_family_1,
                out_seq_len_1,
            )
        elif self.train_type == "MUL":
            return (
                out_low_seq,
                out_masked_seq,
                out_family,
                out_seq_len,
                out_low_seq_1,
                out_masked_seq_1,
                out_family_1,
                out_seq_len_1,
                out_common_index,
                out_common_index_1,
            )
        elif self.train_type == "SSL":
            return (
                out_low_seq,
                out_masked_seq,
                out_family,
                out_seq_len,
                out_low_seq_1,
                out_masked_seq_1,
                out_family_1,
                out_seq_len_1,
                out_common_index,
                out_common_index_1,
                out_SS,
                out_SS_1,
            )
        # elif self.train_type == "SHOW":
        #     return out_low_seq, out_family, out_seq_len, out_SS
        else:
            return out_low_seq, out_family, out_seq_len


def onehot_seq(gapped_seq, pad_max_length):
    gapped_seq = [
        list(
            i.translate(
                str.maketrans(
                    {"-": "0", ".": "0", "A": "1", "U": "1", "G": "1", "C": "1"}
                )
            )
        )
        for i in gapped_seq
    ]
    gapped_seq = [list(map(lambda x: int(x), s)) for s in gapped_seq]
    gapped_seq = np.array(
        [np.pad(s, ((0, pad_max_length - len(s)))) for s in gapped_seq]
    )
    return gapped_seq


def kmer(seqs, k=1):
    kmer_seqs = []
    for seq in seqs:
        kmer_seq = []
        for i in range(len(seq)):
            if i <= len(seq) - k:
                kmer_seq.append(seq[i : i + k])
        kmer_seqs.append(kmer_seq)
    return kmer_seqs


def mask(seqs, rate=0.2, mag=1):
    masked_seq = []
    label = []
    for i in range(mag):
        seqs2 = copy.deepcopy(seqs)
        for s in seqs2:
            label.append(copy.copy(s))
            mask_num = int(len(s) * rate)
            all_change_index = np.array(random.sample(range(len(s)), mask_num))
            mask_index, base_change_index = np.split(
                all_change_index, [int(all_change_index.size * 0.90)]
            )
            for i in list(mask_index):
                s[i] = "MASK"
            for i in list(base_change_index):
                s[i] = random.sample(("A", "U", "G", "C"), 1)[0]
            masked_seq.append(s)
    return masked_seq, label


def convert(seqs, kmer_dict, max_length):
    seq_num = []
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        convered_seq = [kmer_dict[i] for i in s] + [0] * (max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num


def make_dict(k=3):
    # seq to num
    l = ["A", "U", "G", "C"]
    kmer_list = ["".join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i + 1 for i, kmer in enumerate(kmer_list)}
    return dic
