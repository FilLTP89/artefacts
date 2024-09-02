import os
import torch
import random 
import numpy as np
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl
import matplotlib.pyplot as plt

CPKT_PATH = "/gpfs/users/gabrielihu/saved_model/blooming-rain-42/best_model.ckpt"


def save_image(
        x, preds,y, path,
    ):
    cmap = plt.cm.gray
    x = x.cpu().numpy()
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()
    plt.imsave(
        path + "_original_image" + ".png",
        x,
        cmap=cmap,
    )
    plt.imsave(
        path  + "_predicted_image" + ".png",
        preds,
        cmap=cmap,
    )
    plt.imsave(
        path + "_ground_truth_image" + ".png",
        y,
        cmap=cmap,
    )

    




def load_model(checkpoint_path, *args, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AttentionMEDGAN.load_from_checkpoint(checkpoint_path=checkpoint_path, filters =  [8,16,32, 64,128,256,512])
    model.eval()
    return model


def load_module(*args, **kwargs):
    module = Datav2Module(*args, **kwargs)
    module.setup()
    return module

def generate_images(model, 
                    dataloader, 
                    saving_path,
                    device = "cpu",
                    run_name = "run_name"
                    ):
    for idx, batch in enumerate(dataloader):
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            generated = model(x)
        for i in range(generated.size(0)):
            save_image(
                x[i], generated[i], y[i], 
                path = saving_path + {run_name}
            )

        
        
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saving_path = "/gpfs/workdir/gabrielihu/artefacts/generated_images/torch/"
    run_name = None
    acquisition_number = 2
    model = load_model(
        checkpoint_path=CPKT_PATH,
    )
    model = model.to(device)
    print("Model loaded")
    module = load_module()
    acquisition = module.get_acquisition(acquisition_number)
    generate_images(model = model, 
                    dataloader = acquisition, 
                    saving_path = saving_path,
                     run_name = run_name
                    )
    print("Images generated")

if __name__ == "__main__":
    main()