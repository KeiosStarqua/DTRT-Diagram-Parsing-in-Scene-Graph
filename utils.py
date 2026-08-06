import torch


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        b, n, c = x.size()
        idx = idx.unsqueeze(-1).expand(-1, -1, c)
        return torch.gather(x, 1, idx)
    if len(x.size()) == 2:
        b, n = x.size()
        return torch.gather(x, 1, idx)
    raise NotImplementedError("batch_index_select expects a 2D or 3D tensor")


def batch_index_fill(x, x1, x2, idx1, idx2):
    if len(x.size()) == 3:
        c = x.size(-1)
        x = x.scatter(1, idx1.unsqueeze(-1).expand(-1, -1, c), x1)
        x = x.scatter(1, idx2.unsqueeze(-1).expand(-1, -1, c), x2)
        return x
    if len(x.size()) == 2:
        x = x.scatter(1, idx1, x1)
        x = x.scatter(1, idx2, x2)
        return x
    raise NotImplementedError("batch_index_fill expects a 2D or 3D tensor")
