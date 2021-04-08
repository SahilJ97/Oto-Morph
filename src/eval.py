import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import OtoMangueanDataset
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to model', required=True)
parser.add_argument('--beam_size', type=int, required=True)
args = vars(parser.parse_args())

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
print(f"Using device {DEVICE}")


def evaluate():
    test_loader = DataLoader(test_set, batch_size=1, drop_last=True)
    n_correct = 0
    for batch_index, batch in enumerate(test_loader):
        input_dict, labels = batch
        for k, v in input_dict.items():
            input_dict[k] = v.to(DEVICE)
        _, label_indices = torch.max(labels, dim=-1)
        label_indices = label_indices.to(DEVICE)
        outputs = model(input_dict)
        _, output_indices = torch.max(outputs, dim=-1)
        if torch.equal(label_indices, output_indices):
            n_correct += 1
        print("\r" + "                                         ", end="")
        print(f"\rAccuracy: {n_correct / (batch_index + 1)}", end="")


if __name__ == "__main__":
    model = torch.load(args["model"], map_location=DEVICE)
    model.eval()
    model.decoder.beam_size = args["beam_size"]
    test_set = OtoMangueanDataset(glob('../data/*.tst'))
    print("Evaluating...\n")
    with torch.no_grad():
        evaluate()
