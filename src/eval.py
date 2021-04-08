import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import OtoMangueanDataset
from glob import glob

EPSILON = 1e-9

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to model', required=True)
parser.add_argument('--beam_size', type=int, required=True)
args = vars(parser.parse_args())

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
print(f"Using device {DEVICE}")


def evaluate():
    test_loader = DataLoader(test_set, batch_size=1, drop_last=True, shuffle=True)
    n_correct = 0
    n_langs = len(test_set.language_to_index)
    correct_by_lang, total_by_lang = [0]*n_langs, [0]*n_langs
    for batch_index, batch in enumerate(test_loader):
        if not isinstance(batch, list):
            continue
        input_dict, labels = batch
        for k, v in input_dict.items():
            input_dict[k] = v.to(DEVICE)
        lang_index = input_dict["language"][0].item()
        _, label_indices = torch.max(labels, dim=-1)
        label_indices = label_indices.to(DEVICE)
        outputs = model(input_dict)
        _, output_indices = torch.max(outputs, dim=-1)

        # Truncate at first PAD token
        if test_set.pad_index in output_indices.tolist()[0]:
            truncate_at = output_indices.tolist()[0].index(test_set.pad_index)
            output_indices = output_indices[:, :truncate_at]
        if test_set.pad_index in label_indices.tolist()[0]:
            truncate_at = label_indices.tolist()[0].index(test_set.pad_index)
            label_indices = label_indices[:, :truncate_at]

        total_by_lang[lang_index] += 1
        if torch.equal(label_indices, output_indices):
            n_correct += 1
            correct_by_lang[lang_index] += 1
        print("\r" + "                                         ", end="")
        print(f"\rAccuracy: {n_correct / (batch_index + 1)}", end="")
        if batch_index % 100 == 99:
            print(
                "\nLanguage-level accuracies: ",
                [correct_by_lang[i]/(total_by_lang[i] + EPSILON) for i in range(n_langs)],
                "\n"
            )


if __name__ == "__main__":
    model = torch.load(args["model"], map_location=DEVICE)
    model.eval()
    model.decoder.beam_size = args["beam_size"]
    test_set = OtoMangueanDataset(glob('../data/*.tst'))
    print("Evaluating...\n")
    with torch.no_grad():
        evaluate()
