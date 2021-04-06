"""Written by Sahil Jayaram"""

import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import OtoMangueanDataset, CHARACTER_SEQUENCE_LENGTH
from src.model import RNN
from glob import glob
from entmax import entmax15_loss

parser = argparse.ArgumentParser()
parser.add_argument('--lang_embeds', help='Initialization scheme for language embeddings (e.g. "random")',
                    required=True)
parser.add_argument('--embed_size', help='Encoder/decoder size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--lr', help='Learning rate', type=float, required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--beam_size', type=int, default=10)
parser.add_argument('--dropout', type=float, default=.2)
args = vars(parser.parse_args())

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
print(f"Using device {DEVICE}")

MAX_DEV_CHARACTERS = 10000  # due to time/memory constraints, a limited number of dev examples are used for validation


def train():
    train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    dev_loader = DataLoader(train_set, batch_size=1, shuffle=False, drop_last=True)  # batch size = 1 for beam search
    epoch_char_accuracies = []
    for epoch in range(args["epochs"]):
        model.train()  # train mode (use dropout)
        print(f"Beginning epoch {epoch}...")
        running_correctness_loss = 0.
        """for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_dict, labels = batch
            for k, v in input_dict.items():
                input_dict[k] = v.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(input_dict, labels=labels)  # use teacher forcing
            _, label_indices = torch.max(labels, dim=-1)
            loss = torch.mean(
                torch.stack([entmax15_loss(outputs[i], label_indices[i]) for i in range(len(outputs))])
            )
            if batch_index == 500:
                print("Sample of model predictions vs. gold labels:")
                print(torch.max(outputs[:4], dim=-1)[1])
                print(label_indices[:4])
            loss.retain_grad()
            loss.backward()
            optimizer.step()
            running_correctness_loss += loss.item()

            # Print running losses every 20 batches
            if batch_index % 50 == 0:
                print(f"Epoch {epoch} iteration {batch_index}")
                print(f"\tRunning loss: {running_correctness_loss / (batch_index + 1)}")"""
        # Validate
        print("Validating...")
        model.eval()  # eval mode (no dropout)
        with torch.no_grad():
            label_indices, outputs = [], []
            for batch_index, batch in enumerate(dev_loader):
                if len(label_indices) * CHARACTER_SEQUENCE_LENGTH > MAX_DEV_CHARACTERS:
                    break
                input_dict, labels = batch
                for k, v in input_dict.items():
                    input_dict[k] = v.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs.append(model(input_dict).cpu())  # don't use teacher forcing
                label_indices.append(torch.max(labels, dim=-1)[1].cpu())
            print([torch.max(outputs[i], dim=-1)[1] for i in range(3)])
            print(label_indices[:3])
            label_indices = torch.cat(label_indices, dim=0)
            _, output_indices = torch.max(torch.stack(outputs), dim=-1)
            word_accuracy = sum(
                [torch.equal(output_indices[i], label_indices[i]) for i in range(len(label_indices))]
            ) / len(label_indices)
            outputs = torch.cat(outputs, dim=0)
            label_indices = torch.flatten(label_indices)
            outputs = torch.flatten(outputs, 0, 1)
            loss = torch.mean(entmax15_loss(outputs, label_indices)).item()
            output_indices = torch.flatten(output_indices)
            char_accuracy = torch.sum(output_indices == label_indices) / len(label_indices)
            print(f"\tLoss: {loss}")
            print(f"\tCharacter-level accuracy: {char_accuracy}")
            print(f"\tWord-level accuracy: {word_accuracy}")
            if len(epoch_char_accuracies) > 0 and char_accuracy > max(epoch_char_accuracies):
                torch.save(model, f"{args['model_name']}.pt")
            epoch_char_accuracies.append(char_accuracy)


if __name__ == "__main__":
    print("Loading data...")
    train_set = OtoMangueanDataset(glob('../data/*.trn'))
    dev_set = OtoMangueanDataset(glob('../data/*.dev'))

    print("Loading model...")
    n_languages = len(list(train_set.language_to_index.keys()))
    init_lang_embeds = [torch.rand(5, device=DEVICE) for _ in range(n_languages)]  # performance was better when I used 4 vs 6?
    model = RNN(
        embed_size=args["embed_size"],
        n_chars=len(list(train_set.character_to_index.keys())),
        n_tags=len(list(train_set.tags_to_index.keys())),
        init_lang_embeds=init_lang_embeds,
        beam_size=args["beam_size"],
        dropout=args["dropout"]
    )
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    train()
