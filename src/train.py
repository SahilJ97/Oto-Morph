import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import OtoMangueanDataset
from src.model import RNN
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--lang_embeds', help='Initialization scheme for language embeddings (e.g. "random")',
                    required=True)
parser.add_argument('--embed_size', help='Encoder/decoder size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--lr', help='Learning rate', type=float, required=True)
args = vars(parser.parse_args())


def train():
    train_loader =  DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    dev_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    for epoch in range(args["epochs"]):
        print(f"Beginning epoch {epoch}...")
        for batch_index, batch in enumerate(train_loader):
            input_dict, labels = batch
            outputs = model(input_dict)
            print(outputs)
            continue


if __name__=="__main__":
    print("Loading data...")
    train_set = OtoMangueanDataset(glob('../data/*.trn'))
    dev_set = OtoMangueanDataset(glob('../data/*.dev'))

    print("Loading model...")
    n_languages = len(list(train_set.language_to_index.keys()))
    init_lang_embeds = torch.stack([torch.rand(4) for _ in range(n_languages)])
    model = RNN(
        embed_size=args["embed_size"],
        n_chars=len(list(train_set.character_to_index.keys())),
        n_tags=len(list(train_set.tags_to_index.keys())),
        init_lang_embeds=init_lang_embeds
    )
    train()
