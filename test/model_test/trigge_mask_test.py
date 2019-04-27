import torch

batch_size = 3
max_seq_len = 10
embedding_dim = 5

embedding = torch.randint(0,100, (batch_size,max_seq_len, embedding_dim)).type(torch.float32)
mask = [
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
]
ts_mask = torch.Tensor(mask)

trigger_embedding = ts_mask.unsqueeze(-1).expand_as(embedding).type_as(embedding).mul(embedding).mean(dim=(1,))
