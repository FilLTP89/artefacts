import torch
import torch.nn as nn
from diffusers import DDPMScheduler,DDIMScheduler
from einops import rearrange, repeat
import pytorch_lightning as pl
import torch.nn.functional as F
from diffusers import UNet2DModel
from torchsummary import summary
from tqdm import tqdm
import bitsandbytes as bnb
from torch.profiler import profile, record_function, ProfilerActivity

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
                in_channels: int = 1,
                learning_rate: float = 3e-4,
                prediction_type = "epsilon",
                training = True,
                num_steps = 5,
                layers_per_block = 1,
                *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet2DModel(
            in_channels=in_channels*2,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            add_attention=True,
            #block_out_channels=(64, 128, 256, 384),
            #attention_head_dim=4,
            #norm_num_groups=16,
        )
        self.learning_rate = learning_rate
        self.prediction_type = prediction_type
        self.training = training
        self.num_steps = num_steps
        self.n_training_steps = num_steps   
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_training_steps,
            beta_schedule="linear",
            prediction_type=self.prediction_type
        )

    def forward(self, noisy_sample, x, t):
        input = torch.cat([noisy_sample, x], dim=1)
        return self.model(input, t).sample

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size, channels, height, width = y.shape
        
        # Move to CPU to save GPU memory
        noise = torch.rand_like(y, device='cpu').to(self.device)
        timesteps = torch.randint(0, self.n_training_steps, (batch_size,), device='cpu').to(self.device).long()

        noisy_sample = self.noise_scheduler.add_noise(y, noise, timesteps)
        
        # Use automatic mixed precision
        with torch.amp.autocast("cuda"):
            predicted_noise = self(noisy_sample=noisy_sample, x=x, t=timesteps)
            
            if self.prediction_type == "epsilon":
                loss = F.mse_loss(predicted_noise, noise)
            elif self.prediction_type == "sample":
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].to(self.device)
                alpha_t = alpha_t.view(batch_size, 1, 1, 1)
                snr_weights = alpha_t / (1 - alpha_t)
                loss = (snr_weights * F.mse_loss(predicted_noise, y, reduction="none")).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.amp.autocast("cuda")
    def sample(self, x):
        batch_size = x.shape[0]
        xt = torch.rand_like(x, device='cpu').to(self.device)
        timesteps = torch.linspace(0, self.num_steps, self.num_steps, device='cpu').to(self.device)
        timesteps = repeat(timesteps, "i -> i b", b=batch_size)
        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_steps)

        steps = tqdm(range(self.num_steps-1, -1, -1)) if not self.training else range(self.num_steps-1, -1, -1)
        for step in steps:
            with torch.no_grad():
                model_output = self(xt, x, timesteps[step])
                xt = self.noise_scheduler.step(
                    model_output=model_output,
                    timestep=step,
                    sample=xt
                ).prev_sample
            torch.cuda.empty_cache()  # Clear cache after each step
        return xt   
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        xt = self.sample(x)
        MSE = F.mse_loss(y, xt)
        self.log("MSE_loss", MSE, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"MSE": MSE}

    def configure_optimizers(self):
        return bnb.optim.AdamW8bit(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    model = Diffusion_UNET(in_channels=1).to("cuda")
    x = torch.randn(16, 1, 32, 32).to("cuda")
    t = torch.Tensor([16]).to("cuda")
    """
    out = model(x,x,t)
    print(out.shape)
    """
    model.training_step((x,x),0)