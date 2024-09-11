import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from diffusers import UNet2DModel


def _extract_into_tensor(arr, timesteps, broadcast_shape, device):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    arr = arr.to(device)  # Move arr tensor to the appropriate device
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class Diffusion_UNET(pl.LightningModule):
    def __init__(self,
                input_channels: int,
                learning_rate: float,
                device = "cuda",
                prediction_type = "epsilon",):
        super().__init__()

        self.model = UNet2DModel(in_channels=input_channels,
                                 out_channels=input_channels)
        self.learning_rate = learning_rate
        self.device = device
        self.prediction_type = prediction_type


    def forward(self,noisy_sample,x,t):
        input = torch.cat([noisy_sample,x],dim=1)
        return self.model(input,t).sample

    
    def training_step(self, batch, batch_idx):
        x,y = batch
        batch_size, channels, sample_size = y.shape
        noise = torch.randn((batch_size, channels, sample_size))
        timesteps = torch.randint(0, self.n_training_steps, (batch_size,), device = self.device).long()

        noisy_sample = self.noise_scheduler.add_noise(y, noise, timesteps)


        predicted_noise = self.model(noisy_sample,x,timesteps).sample

        if self.prediction_type == "epsilon":
            loss = F.mse_loss(predicted_noise,noise)

        elif self.prediction_type == "sample":
            alpha_t = _extract_into_tensor(
                    self.noise_scheduler.alphas_cumprod, timesteps, (y.shape[0], 1, 1, 1),self.device
                )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * F.mse_loss(
                    predicted_noise, y, reduction="none"
                )  # use SNR weighting from distillation paper
            loss = loss.mean()
        return loss




if __name__ == "__main__":
    model = UNet2DModel(in_channels=1, out_channels=1).to("cuda")
    x = torch.randn(1, 1, 64, 64).to("cuda")
    t = torch.Tensor([1]).to("cuda")
    out = model(x,t).sample
    print(out.shape)