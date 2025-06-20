import yaml
import argparse
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from model import ModelWrapper
from dataset import DataWrapper


def main(config_file):
    # Load configuration
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    data = DataWrapper(
        batch_size=config["batch_size"], vn=config["vn"], ve=config["ve"]
    )

    # Initialize MeshGAT model
    model = ModelWrapper(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        heads=config["heads"],
        edge_dim=config["edge_dim"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        min_lr=config["min_lr"],
        patience=config["lr_decay_patience"],
        rigid_edge_loss=config["rel"],
    )

    # Train model, project name and api token for neptune obtained from env variables
    logger = L.pytorch.loggers.NeptuneLogger()

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath="GNN/Data/saved_models",
                filename=f"{logger.run['sys/name'].fetch()}-{{val_loss_mee:.3f}}",
                monitor="val_loss_mee",
            ),
            EarlyStopping(
                monitor="val_loss_mee", patience=config["early_stop_patience"]
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Geometric Training")
    parser.add_argument(
        "--config",
        type=str,
        default="GNN/config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    main(args.config)
