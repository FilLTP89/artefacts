from data_file.processing_segmentation_torch import SegmentationTorchDataset
from parsing import parse_args, default_config
from torch.optim import Adam
import monai
from transformers import SamModel
import torch
from tqdm import tqdm
from statistics import mean
import faulthandler
#faulthandler.enable()
import warnings
import os
warnings.filterwarnings("ignore") # delete warnings
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp



os.environ['TRANSFORMERS_CACHE'] = "/gpfswork/rech/tui/upz57sx/dossier_hugo/hugginface_cache/"
SAVING_FOLDER = "model/saved_model/segmentation/SAM/"

def train(model, num_epochs, device, train_dataloader, optimizer):
    model = model.to(device)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        bar = tqdm(train_dataloader)
        for batch in bar:
          #  forward pass
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device)
            outputs = model(img)
            # compute loss
            predicted_masks = outputs.logits

            loss = seg_loss(predicted_masks,mask)
            # backward pass (compute gradients of parameters w.r.t. loss)
            loss.backward()
            # optimize
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
            bar.set_postfix(loss = loss.item())
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        torch.save(model.state_dict(), SAVING_FOLDER + f"model_epoch_{epoch}.pt") 

if __name__ == "__main__":
    parse_args()
    if not os.path.exists(SAVING_FOLDER):
        os.makedirs(SAVING_FOLDER)
    module = SegmentationTorchDataset(dataset_path="../data/segmentation", backbone="efficientnet-b0")
    train_dataloader = DataLoader(module, batch_size=16, shuffle=True)
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=2,
    )

    for name, param in model.named_parameters():
        param.requires_grad_(True)


    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, 100, device, train_dataloader, optimizer)
