import torch
import torch.nn.functional as F


def mee_loss(pred, y):
    return F.pairwise_distance(pred, y).mean()


def rigid_edge_loss(pred, edge_index, edge_attr, pos, mse=True, weight=1.0):
    l = edge_attr[:, 0]
    rigid_mask = torch.any(edge_attr[:, 1:] > 0, dim=1)

    v0, v1 = edge_index[:, rigid_mask]
    l = l[rigid_mask]

    # Calculate current lengths for rigid edges
    a = pos[v0] + pred[v0]
    b = pos[v1] + pred[v1]
    l_new = torch.norm(b - a, dim=1)

    loss = F.mse_loss(l_new, l) if mse else F.l1_loss(l_new, l)

    return loss * weight


def losses(pred, y, rigid_mask):
    mee = mee_loss(pred, y)
    mae = F.l1_loss(pred, y)
    mse = F.mse_loss(pred, y)

    soft_mask = rigid_mask == 0
    rigid_mask = ~soft_mask
    rigid_loss = mee_loss(pred[rigid_mask], y[rigid_mask])
    soft_loss = mee_loss(pred[soft_mask], y[soft_mask])

    return mee, mae, mse, rigid_loss, soft_loss
