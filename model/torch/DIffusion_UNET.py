import torch
import torch.nn as nn
from diffusers import DDPMScheduler,DDIMScheduler
from einops import rearrange, repeat
import pytorch_lightning as pl
import torch.nn.functional as F
from diffusers import UNet2DModel,UNet2DConditionModel
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
                num_steps = 10,
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
        noise = torch.randn_like(y).to(self.device)
        timesteps = torch.randint(0, self.n_training_steps, (batch_size,)).to(self.device).long()

        noisy_sample = self.noise_scheduler.add_noise(y, noise, timesteps)
        
        # Use automatic mixed precision
        with torch.amp.autocast("cuda"):
            predicted_noise = self(noisy_sample=noisy_sample, x=x, t=timesteps)
            
            if self.prediction_type == "epsilon":
                loss = F.mse_loss(predicted_noise, noise)
            elif self.prediction_type == "sample":
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps]
                snr_weights = alpha_t / (1 - alpha_t)
                loss = (snr_weights.view(-1, 1, 1, 1) * F.mse_loss(predicted_noise, y, reduction="none")).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    @torch.no_grad()
    def sample(self, x):
        batch_size = x.shape[0]
        xt = torch.randn_like(x, device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_steps)
    
        for t in self.noise_scheduler.timesteps:
            t_input = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            noise_pred = self(noisy_sample=xt, x=x, t=t_input)
            xt = self.noise_scheduler.step(model_output=noise_pred, timestep=t, sample=xt).prev_sample
        
        return xt
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        xt = self.sample(x)
        MSE = F.mse_loss(y, xt)
        self.log("MSE_loss", MSE, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        return {"MSE": MSE}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "MSE_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["noise_scheduler"] = self.noise_scheduler

    def on_load_checkpoint(self, checkpoint):
        self.noise_scheduler = checkpoint["noise_scheduler"]


class ImageToImageDDIMLightningModule(pl.LightningModule):
    def __init__(self, img_size=512, num_channels=1):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        
        # Initialize the UNet2DConditionModel
        self.unet = UNet2DConditionModel(
            sample_size=img_size,
            in_channels=num_channels,
            out_channels=num_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=num_channels * img_size * img_size,  # flattened bad image dimension
        )
        
        # Initialize the DDIM scheduler
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, bad_image, timestep, good_image):
        """
        Forward pass of the model.
        
        Args:
            bad_image (torch.Tensor): The bad image input (B, C, H, W).
            timestep (torch.Tensor): The current timestep in the diffusion process.
            good_image (torch.Tensor): The good image input (B, C, H, W).
        
        Returns:
            torch.Tensor: The predicted noise.
        """
        # Print shape information for debugging
        print(f"bad_image shape: {bad_image.shape}")
        print(f"timestep shape: {timestep.shape}")
        print(f"good_image shape: {good_image.shape}")
        
        # Ensure timestep is a 1D tensor
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # Flatten the bad image to use as condition
        condition = bad_image.view(bad_image.shape[0], -1)
        
        try:
            return self.unet(good_image, timestep, condition).sample
        except ValueError as e:
            print(f"Error in forward pass: {e}")
            print(f"UNet expected shapes: in_channels={self.unet.config.in_channels}, cross_attention_dim={self.unet.config.cross_attention_dim}")
            raise

    def training_step(self, batch, batch_idx):
        bad_images, good_images = batch
        
        # Sample noise to add to the good images
        noise = torch.randn(good_images.shape).to(good_images.device)
        bs = good_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=good_images.device).long()

        # Add noise to the good images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(good_images, noise, timesteps)

        # Get the model prediction for the noise
        noise_pred = self(bad_images, timesteps, noisy_images)

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optm.Adam(self.parameters(), lr=1e-4)

    def sample(self, bad_image, num_inference_steps=50):
        # Ensure bad_image has the correct shape
        if bad_image.dim() == 3:
            bad_image = bad_image.unsqueeze(0)  # Add batch dimension if it's missing
        
        # Start from pure noise
        image = torch.randn(
            (bad_image.shape[0], self.num_channels, self.img_size, self.img_size),
            device=self.device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Scale the noisy image according to the scheduler
            model_input = self.noise_scheduler.scale_model_input(image, t)

            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self(bad_image, t, model_input)

            # Compute the previous noisy sample x_t -> x_t-1
            image = self.noise_scheduler.step(noise_pred, t, image).prev_sample

        # Scale and clip the image to [-1, 1]
        image = torch.clamp(image, 0, 1.0)
        
        return image

if __name__ == "__main__":
    """
    model = Diffusion_UNET(in_channels=1).to("cuda")
    x = torch.randn(16, 1, 32, 32).to("cuda")
    t = torch.Tensor([16]).to("cuda")
    out = model(x,x,t)
    print(out.shape)
    """
    model = ImageToImageDDIMLightningModule().to("cuda")
    summary(model, [(1, 1, 512, 512), (1,), (1, 1, 512, 512)])
    bad_image = torch.randn(1, 1, 512, 512).to("cuda")
    t = torch.Tensor([1]).to("cuda")
    good_image = torch.randn(1, 1, 512, 512).to("cuda")
    out = model(bad_image, t, good_image)
