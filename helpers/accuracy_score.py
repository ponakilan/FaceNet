import numpy as np


def accuracy_score(anchor_embed, pos_embed, neg_embed, threshold=0.3):
    anchor_embed, pos_embed, neg_embed = anchor_embed.cpu().detach().numpy(), pos_embed.cpu().detach().numpy(), neg_embed.cpu().detach().numpy()
    prediction = []
    for i in range(anchor_embed.shape[0]):
        pos_dist = abs(np.linalg.norm(anchor_embed[i]) - np.linalg.norm(pos_embed[i]))
        neg_dist = abs(np.linalg.norm(anchor_embed[i]) - np.linalg.norm(neg_embed[i]))

        if pos_dist < neg_dist:
            prediction.append(1)
        else:
            prediction.append(0)

    return pos_dist, neg_dist, prediction.count(1)/len(prediction)
