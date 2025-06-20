from lightning.pytorch import Trainer
from dataset import DataWrapper
from model import ModelWrapper

# Load data
data = DataWrapper(vn=True)

# Load the checkpoint
checkpoint_path = "GNN/Data/saved_models/GNN-371-val_loss_mee=0.253.ckpt"

model = ModelWrapper.load_from_checkpoint(checkpoint_path)
trainer = Trainer()

# Run the test
test_results = trainer.test(model, datamodule=data, ckpt_path=checkpoint_path)
