import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import OtoMangueanDataset
from src.model import RNN
from glob import glob
from external.fyl_pytorch import SparsemaxLoss

parser = argparse.ArgumentParser()
parser.add_argument('--lang_embeds', help='Initialization scheme for language embeddings (e.g. "random")',
                    required=True)
parser.add_argument('--embed_size', help='Encoder/decoder size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--lr', help='Learning rate', type=float, required=True)
args = vars(parser.parse_args())


def train():
    train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    dev_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    for epoch in range(args["epochs"]):
        print(f"Beginning epoch {epoch}...")
        running_correctness_loss = 0.
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_dict, labels = batch
            outputs = model(input_dict, labels=labels)  # use teacher forcing on train batches (labels != None)
            _, label_indices = torch.max(labels, dim=-1)
            loss = torch.mean(
                torch.stack([correctness_loss(outputs[i], label_indices[i]) for i in range(len(outputs))])
            )
            running_correctness_loss += loss.item()
            loss.backward()

            # Print running losses every 20 batches
            if batch_index % 50 == 0:
                print(f"Epoch {epoch} iteration {batch_index}")
                print(f"\tRunning loss: {running_correctness_loss / (batch_index + 1)}")


if __name__ == "__main__":
    correctness_loss = SparsemaxLoss()

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    train()
