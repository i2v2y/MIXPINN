import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, JumpingKnowledge, SAGEConv
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from dataset import DataWrapper
from model import ModelWrapper


# cnofig 3
class PhysGNN(nn.Module):

    def __init__(self, input_dim=14, hidden_dim=112, output_dim=3):
        super(PhysGNN, self).__init__()

        self.conv1 = GraphConv(input_dim, hidden_dim, aggr="add")
        self.conv2 = GraphConv(hidden_dim, hidden_dim, aggr="add")
        self.conv3 = GraphConv(hidden_dim, hidden_dim, aggr="max")
        self.conv4 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv5 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv6 = SAGEConv(hidden_dim, hidden_dim, aggr="max")

        self.jk1 = JumpingKnowledge("lstm", hidden_dim, 3)
        self.jk2 = JumpingKnowledge("lstm", hidden_dim, 3)

        self.lin1 = torch.nn.Linear(hidden_dim, 63)
        self.lin2 = torch.nn.Linear(63, 3)

        self.active1 = nn.PReLU(hidden_dim)
        self.active2 = nn.PReLU(hidden_dim)
        self.active3 = nn.PReLU(hidden_dim)
        self.active4 = nn.PReLU(hidden_dim)
        self.active5 = nn.PReLU(hidden_dim)
        self.active6 = nn.PReLU(hidden_dim)
        self.active7 = nn.PReLU(63)

    def forward(self, data):
        x, edge_index, edge_weight = (data.x, data.edge_index, data.edge_attr[:, 0])
        edge_weight = 1 / edge_weight
        edge_weight = edge_weight.float()

        x = self.conv1(x, edge_index, edge_weight)
        x = self.active1(x)
        xs = [x]

        x = self.conv2(x, edge_index, edge_weight)
        x = self.active2(x)
        xs += [x]

        x = self.conv3(x, edge_index, edge_weight)
        x = self.active3(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk1(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.conv4(x, edge_index)
        x = self.active4(x)
        xs = [x]

        x = self.conv5(x, edge_index)
        x = self.active5(x)
        xs += [x]

        x = self.conv6(x, edge_index)
        x = self.active6(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk2(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.lin1(x)
        x = self.active7(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)

        return x


def main():
    # Prepare data
    data = DataWrapper(batch_size=8)

    # Initialize model
    model = ModelWrapper(PhysGNN, lr=0.005, patience=5, min_lr=1e-8)

    logger = L.pytorch.loggers.NeptuneLogger()

    trainer = L.Trainer(
        max_epochs=500,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath="GNN/Data/saved_models",
                filename=f"{logger.run['sys/name'].fetch()}-physgnn-{{val_loss_mee:.4f}}",
                monitor="val_loss_mee",
            ),
            EarlyStopping(monitor="val_loss_mee", patience=15),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
