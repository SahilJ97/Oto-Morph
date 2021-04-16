'''Written by Diana Abagyan'''

import pickle
import json
import torch
import matplotlib.pyplot as plt


#epochs = 20
threshold = .005
with open("../dictionaries/language_to_index.json") as f:
    lang_to_ind = json.load(f)


def loss_fn(lang_embeds, true_dists):
    loss_sum = 0
    for l1, l2, dist in true_dists:
        v1 = lang_embeds[lang_to_ind[l1]]
        v2 = lang_embeds[lang_to_ind[l2]]
        loss_sum += abs(dist - torch.linalg.norm(v1 - v2, 2))

    return loss_sum


if __name__ == '__main__':
    with open("../linguistic_distance/leven_dist_def.pkl", 'rb') as f:
        true_dists = pickle.load(f)

    # initialize embeddings
    lang_embeds = [torch.rand(5, requires_grad=True) for _ in range(10)]

    # optimize
    optimizer = torch.optim.SGD(lang_embeds, lr=0.01)
    loss_record = []
    optimize = True
    epochs = 0
    while optimize:
        epochs += 1
        optimizer.zero_grad()
        loss = loss_fn(lang_embeds, true_dists)
        loss_record.append(loss.item())
        loss.backward()
        optimizer.step()

        if len(loss_record) > 2:    # stop if difference between epochs is below threshold
            if (loss_record[-2] - loss_record[-1]) < threshold:
                optimize = False

    # plot loss
    plt.plot([i for i in range(epochs)], loss_record)
    plt.show()

    # save embeddings
    torch.save(lang_embeds, "../linguistic_distance/language_embeddings_def.pt")