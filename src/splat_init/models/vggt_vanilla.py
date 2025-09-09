from pathlib import Path

import torch as th
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX


class VggtPerspectiveTransform(LightningModule):
    def __init__(self, model_url: Path = Path("facebook/VGGT-1B")) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = VGGT.from_pretrained(model_url)
        

    def (self, images: th.Tensor) -> th.Tensor:
        pass
        


    def forward(self, images: th.Tensor) -> th.Tensor:
        return self.model(images)


    def training_step(self, batch, batch_idx):
        pass
    
    
    def validation_step(self, batch, batch_idx):
        images, _ = batch
        outputs = self(images)
        loss = th.nn.functional.mse_loss(outputs, images)
        self.log(f"{VALIDATION_PREFIX}_loss", loss)
        return loss

    
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        
        
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=None,
        )




if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    dtype = th.bfloat16 if th.cuda.is_available() else th.float16
    
    model = VggtPerspectiveTransform().to(device)
    with th.no_grad():
        with th.amp.autocast(device, dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model.forward(images)