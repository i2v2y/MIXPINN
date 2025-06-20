import os
import numpy as np
import pyvista as pv

import torch
import lightning as L
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import one_hot
from torch_geometric.transforms import FaceToEdge
from torch_geometric.loader import DataLoader


class MeshDataset(Dataset):

    def __init__(
        self,
        root: str = "GNN/Data",
        split: str = "train",
        vn: bool = False,
        ve: bool = False,
    ):
        super(MeshDataset, self).__init__(root)

        if split == "train":
            self.xy = torch.load(self.processed_paths[0])
        elif split == "val":
            self.xy = torch.load(self.processed_paths[1])
        elif split == "test":
            self.xy = torch.load(self.processed_paths[2])
        else:
            self.xy = torch.load(self.processed_paths[6])

        if vn:
            self.data = torch.load(self.processed_paths[5])
        elif ve:
            self.data = torch.load(self.processed_paths[4])
        else:
            self.data = torch.load(self.processed_paths[3])

    @property
    def raw_paths(self):
        return [
            "Simulation/data/inputs/probe_positions.txt",
            "Simulation/data/outputs/batch",
            "Simulation/data/mesh/body.vtu",
            "GNN/Data/raw/x0z0y105_0.npz",
        ]

    @property
    def processed_file_names(self):
        return [
            "train.pt",
            "val.pt",
            "test.pt",
            "data.pt",
            "data_ve.pt",
            "data_vn.pt",
            "predict.pt",
        ]

    def process(self):
        probe_positions = np.loadtxt(self.raw_paths[0], dtype=int)
        probe_positions = [f"x{x}z{z}" for x, y, z in probe_positions]

        train_pos, val_pos, test_pos = random_split(
            probe_positions,
            [0.7, 0.2, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

        train, val, test = [], [], []

        with os.scandir(self.raw_paths[1]) as files:
            for file in files:
                data = np.load(file.path)
                data = {
                    "X": torch.tensor(data["X"], dtype=torch.float),
                    "Y": torch.tensor(data["Y"], dtype=torch.float),
                }

                # Determine which split this file belongs to
                xz = file.name.split("y")[0]
                if xz in train_pos:
                    train.append(data)
                elif xz in val_pos:
                    val.append(data)
                else:  # xz in test_pos
                    test.append(data)

        torch.save(train, self.processed_paths[0])
        torch.save(val, self.processed_paths[1])
        torch.save(test, self.processed_paths[2])

        data = np.load(self.raw_paths[3])
        data = {
            "X": torch.tensor(data["X"], dtype=torch.float),
            "Y": torch.tensor(data["Y"], dtype=torch.float),
        }
        torch.save([data], self.processed_paths[6])

        mesh = pv.read(self.raw_paths[2])
        pos = torch.tensor(mesh.points, dtype=torch.float)
        pos_spherical = self.cartesian_to_spherical(pos)
        rigid_mask = torch.tensor(mesh.point_data["rigid"], dtype=torch.long)
        rigid_one_hot = one_hot(rigid_mask, num_classes=max(rigid_mask) + 1)[:, 1:]

        # Get cell connectivity, exclude 1st col cell type
        cells = mesh.cells.reshape(-1, 5)[:, 1:]
        data = Data(
            x_base=torch.hstack([pos, pos_spherical, rigid_one_hot]),
            pos=pos,
            rigid_mask=rigid_mask,
            face=torch.tensor(cells, dtype=torch.long).t(),
        )

        # Generate edge index, bi-directional
        data = FaceToEdge()(data)

        # use length of edge and one hot encoding of the rigid group as edge features
        i, j = data.edge_index
        l = torch.norm(pos[j] - pos[i], dim=1)
        edge_mask = torch.where(
            rigid_one_hot[i] == rigid_one_hot[j],
            rigid_one_hot[i],
            torch.zeros_like(rigid_one_hot[i]),
        )
        data.edge_attr = torch.hstack([l.unsqueeze(1), edge_mask])
        torch.save(data, self.processed_paths[3])

        torch.save(self.add_virtual_edge(data.clone()), self.processed_paths[4])
        torch.save(self.add_virtual_node(data.clone()), self.processed_paths[5])

    def len(self):
        return len(self.xy)

    def get(self, idx):
        self.data.x = self.xy[idx]["X"]
        self.data.y = self.xy[idx]["Y"]

        if hasattr(self.data, "vn_x"):
            self.data.x = torch.cat([self.data.x, self.data.vn_x])
            rigid_mask = self.data.rigid_mask[: -self.data.vn_x.shape[0]]

            vn_y = []
            for i in range(rigid_mask.max()):
                rigid_nodes = (rigid_mask == i + 1).nonzero().squeeze(dim=1)

                # use mean y of all rigid nodes
                y_mean = self.data.y[rigid_nodes].mean(dim=0)
                vn_y.append(y_mean)

            self.data.y = torch.cat([self.data.y, torch.stack(vn_y)])

        self.data.x = torch.hstack([self.data.x, self.data.x_base])
        return self.data

    def cartesian_to_spherical(self, pos):
        r = torch.norm(pos, dim=1)
        theta = torch.atan2(pos[:, 1], pos[:, 0])  # azimuthal angle
        phi = torch.acos(pos[:, 2] / r)  # polar angle
        return torch.stack((r, theta, phi), dim=1)

    def add_virtual_node(self, data):
        pos = data.pos
        rigid_mask = data.rigid_mask
        rigid_one_hot = torch.eye(rigid_mask.max())

        # Create virtual nodes which connect to all rigid points in the same part
        vn_pos = []
        vn_edge_index = []
        vn_edge_attr = []

        for i in range(rigid_mask.max()):
            vn_id = pos.shape[0] + i

            rigid_nodes = (rigid_mask == i + 1).nonzero().squeeze(dim=1)

            # use mean position of all rigid nodes
            pos_mean = pos[rigid_nodes].mean(dim=0)
            vn_pos.append(pos_mean)

            # connect vn to all rigid nodes
            for node in rigid_nodes:
                vn_edge_index.extend(
                    [torch.tensor([node, vn_id]), torch.tensor([vn_id, node])]
                )

                edge_attr = torch.hstack(
                    [torch.norm(pos_mean - pos[node]), rigid_one_hot[i]]
                )
                vn_edge_attr.extend([edge_attr, edge_attr])  # Add for both directions

        vn_pos = torch.stack(vn_pos)
        vn_pos_spherical = self.cartesian_to_spherical(vn_pos)

        vn_x_base = torch.hstack(
            [
                vn_pos,  # Shape: [5, 3]
                vn_pos_spherical,  # Shape: [5, 3]
                rigid_one_hot,  # Shape: [5, 5]
            ]
        )

        vn_edge_index = torch.stack(vn_edge_index).t().contiguous()
        vn_edge_attr = torch.stack(vn_edge_attr)

        data.pos = torch.cat([data.pos, vn_pos])
        data.rigid_mask = torch.cat(
            [data.rigid_mask, torch.arange(1, rigid_mask.max() + 1)]
        )
        data.x_base = torch.cat([data.x_base, vn_x_base])
        data.edge_index = torch.cat([data.edge_index, vn_edge_index], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, vn_edge_attr])

        data.vn_x = torch.zeros_like(vn_pos)
        return data

    def add_virtual_edge(self, data):
        pos = data.pos
        rigid_mask = data.rigid_mask
        rigid_one_hot = torch.eye(rigid_mask.max())
        edges = {frozenset(edge.tolist()) for edge in data.edge_index.t()}

        ve_index = []
        ve_attr = []
        for i in range(rigid_mask.max()):
            rigid_nodes = (rigid_mask == i + 1).nonzero().squeeze(dim=1)
            ymax_node = rigid_nodes[torch.argmax(pos[rigid_nodes, 1])]

            # Create edges from all other nodes in the rigid part to the highest y-position node
            for node in rigid_nodes:
                if node == ymax_node:
                    continue
                if frozenset(torch.tensor([node, ymax_node]).tolist()) in edges:
                    continue
                # Add edges in both directions
                ve_index.extend(
                    [torch.tensor([node, ymax_node]), torch.tensor([ymax_node, node])]
                )
                edge_attr = torch.hstack(
                    (torch.norm(pos[node] - pos[ymax_node]), rigid_one_hot[i])
                )
                ve_attr.extend([edge_attr, edge_attr])  # Add for both directions

        edge_index = torch.stack(ve_index).t().contiguous()
        edge_attr = torch.stack(ve_attr)

        data.edge_index = torch.cat([data.edge_index, edge_index], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, edge_attr])

        return data


class DataWrapper(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "GNN/Data",
        batch_size: int = 1,
        ve: bool = False,
        vn: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train = MeshDataset(root=data_dir, split="train", ve=ve, vn=vn)
        self.val = MeshDataset(root=data_dir, split="val", ve=ve, vn=vn)
        self.test = MeshDataset(root=data_dir, split="test", ve=ve, vn=vn)
        self.predict = MeshDataset(root=data_dir, split="predict", ve=ve, vn=vn)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val)

    def test_dataloader(self):
        return DataLoader(self.test)

    def predict_dataloader(self):
        return DataLoader(self.predict)


def main():
    dataset = MeshDataset(vn=True, split="predict")
    print(dataset)


if __name__ == "__main__":
    main()
