from data_file.SAMdataset import SAMModule
from parsing import parse_args, default_config
from torch.optim import Adam
import monai
from transformers import SamModel
import torch
from tqdm import tqdm
from statistics import mean
import faulthandler

#faulthandler.enable()

def train(model, num_epochs, device, train_dataloader, optimizer):
    model = model.to(device)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
          # forward pass
          pixel_values = batch["pixel_values"].to(device)
          input_boxes = batch["input_boxes"].to(device)
          outputs = model(pixel_values=pixel_values,
                          input_boxes=input_boxes,
                          multimask_output=False)
          # compute loss

          predicted_masks = outputs.pred_masks.squeeze(1)
          ground_truth_masks = batch["ground_truth_mask"].float().to(device)
          loss = seg_loss(predicted_masks, ground_truth_masks)

          # backward pass (compute gradients of parameters w.r.t. loss)
          optimizer.zero_grad()
          loss.backward()

          # optimize
          optimizer.step()
          epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    

if __name__ == "__main__":
    parse_args()
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = SAMModule(batch_size= 1)
    train(model, 10, device, module.train_loader, optimizer)