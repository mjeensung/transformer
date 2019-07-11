import torch

mask = torch.zeros((4,4))
for i, row in enumerate(mask):
    for j, column in enumerate(row):
        if i < j:
            mask[i][j] = -float("inf")

print(mask)