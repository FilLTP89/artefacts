import monai
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import SamModel
import lightning as L


class SAM_model(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = SamModel.from_pretrained("wanglab/medsam-vit-base")

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        #Try DiceFocalLoss, FocalLoss, DiceCELoss
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def configure_optimizer(self):
        return Adam(self.model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"],
                      input_boxes=batch["input_boxes"],
                      multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float()
        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pass
    pass

    def __call__(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = SAM_model()
    x = torch.rand((1, 1, 1024, 1024))
    y = model(x,)
    predicted_masks = y.pred_masks.squeeze(1)
    print(predicted_masks.shape)   