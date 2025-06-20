import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.models import GAT

import time

from loss import *


class MeshGAT(nn.Module):

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 256,
        output_dim: int = 3,
        heads: int = 2,
        edge_dim: int = 6,
        num_layers: int = 8,
    ):
        super(MeshGAT, self).__init__()

        if edge_dim != 0:
            self.gat = GAT(
                input_dim,
                hidden_dim,
                out_channels=output_dim,
                heads=heads,
                edge_dim=edge_dim,
                num_layers=num_layers,
            )
        else:
            self.gat = GAT(
                input_dim,
                hidden_dim,
                out_channels=output_dim,
                heads=heads,
                num_layers=num_layers,
            )

        self.edge_dim = edge_dim

    def forward(self, data):
        if self.edge_dim != 0:
            return self.gat(data.x, data.edge_index, edge_attr=data.edge_attr)
        else:
            return self.gat(data.x, data.edge_index)


class ModelWrapper(L.LightningModule):

    def __init__(
        self,
        model: nn.Module = MeshGAT,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        patience: int = 5,
        min_lr: float = 1e-8,
        rigid_edge_loss: bool = False,
        **kwargs,
    ):
        super(ModelWrapper, self).__init__()

        self.model = model(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_lr = min_lr
        self.rel = rigid_edge_loss
        self.save_hyperparameters()

        # Initialize metrics
        self.test_mee = []
        self.test_mae = []
        self.test_mse = []
        self.test_rigid = []
        self.test_soft = []
        self.test_edge = []

        self.t = []

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = mee_loss(out, batch.y)

        if self.rel:
            loss += rigid_edge_loss(out, batch.edge_index, batch.edge_attr, batch.pos)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        mee, mae, mse, rigid_loss, soft_loss = losses(out, batch.y, batch.rigid_mask)
        rel = rigid_edge_loss(out, batch.edge_index, batch.edge_attr, batch.pos)

        self.log_dict(
            {
                "val_loss_mee": mee,
                "val_loss_mae": mae,
                "val_loss_mse": mse,
                "val_loss_rigid": rigid_loss,
                "val_loss_soft": soft_loss,
                "val_loss_edge": rel,
            }
        )
        return mee

    def test_step(self, batch, batch_idx):
        start = time.time()
        out = self(batch)
        end = time.time()
        self.t.append(end - start)

        mee, mae, mse, rigid_loss, soft_loss = losses(out, batch.y, batch.rigid_mask)
        rel = rigid_edge_loss(
            out, batch.edge_index, batch.edge_attr, batch.pos, mse=False
        )

        self.log_dict(
            {
                "test_mee_mean": mee,
                "test_mae_mean": mae,
                "test_mse_mean": mse,
                "test_rigid_mean": rigid_loss,
                "test_soft_mean": soft_loss,
                "test_edge_mean": rel,
            }
        )

        # acculmulate test metrics
        self.test_mee.append(mee)
        self.test_mae.append(mae)
        self.test_mse.append(mse)
        self.test_rigid.append(rigid_loss)
        self.test_soft.append(soft_loss)
        self.test_edge.append(rel)
        return mee

    def on_test_epoch_end(self):
        # Report standard deviation of test results
        self.log_dict(
            {
                "test_mee_std": torch.std(torch.tensor(self.test_mee)),
                "test_mae_std": torch.std(torch.tensor(self.test_mae)),
                "test_mse_std": torch.std(torch.tensor(self.test_mse)),
                "test_rigid_std": torch.std(torch.tensor(self.test_rigid)),
                "test_soft_std": torch.std(torch.tensor(self.test_soft)),
                "test_edge_std": torch.std(torch.tensor(self.test_edge)),
                "avg_inference_time": torch.mean(torch.tensor(self.t)),
            }
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print(mee_loss(self(batch), batch.y))
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                patience=self.patience,
                min_lr=self.min_lr,
            ),
            "monitor": "val_loss_mee",  # The metric to monitor
            "interval": "epoch",  # Adjust per epoch
        }
        return [optimizer], [scheduler]
