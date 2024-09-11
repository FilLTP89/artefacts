import torch
from diffusers import DDPMScheduler,DDIMScheduler
from einops import rearrange, repeat
import pytorch_lightning as pl
import torch.nn.functional as F
from diffusers import UNet2DModel
from torchsummary import summary
from tqdm import tqdm

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
                num_steps = 20,
                layers_per_block = 2,
                *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet2DModel(in_channels=in_channels*2,
                                 out_channels=in_channels,
                                 layers_per_block=layers_per_block)
        self.learning_rate = learning_rate
        self.prediction_type = prediction_type
        self.training = training
        self.num_steps = num_steps
        self.n_training_steps = num_steps   
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps= self.n_training_steps,
            beta_schedule="linear",
            prediction_type= self.prediction_type

        )

    def forward(self,noisy_sample,x,t):
        input = torch.cat([noisy_sample,x],dim=1)
        return self.model(input,t).sample

    
    def training_step(self, batch, batch_idx):
        x,y = batch
        batch_size, channels, height, widht = y.shape
        noise = torch.rand_like(y)
        timesteps = torch.randint(0, self.n_training_steps, (batch_size,), device = self.device).long()

        noisy_sample = self.noise_scheduler.add_noise(y, noise, timesteps)

        predicted_noise = self(noisy_sample =noisy_sample,
                                     x = x,
                                     t = timesteps)

        if self.prediction_type == "epsilon":
            loss = F.mse_loss(predicted_noise,noise)

        elif self.prediction_type == "sample":
            alpha_t = _extract_into_tensor(
                    self.noise_scheduler.alphas_cumprod, timesteps, (y.shape[0], 1, 1, 1),self.device
                )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * F.mse_loss(
                    predicted_noise, y, reduction="none"
                ) 
            loss = loss.mean()
        self.log("train_loss",loss, on_step=True, on_epoch=True, prog_bar=True,rank_zero_only=True,sync_dist=True)
        return loss

    @torch.amp.autocast("cuda") 
    def sample(self,x):
        batch_size = x.shape[0]
        xt = torch.rand_like(x)
        timesteps = torch.linspace(0,self.num_steps, self.num_steps).to(self.device)
        timesteps = repeat(timesteps, "i -> i b", b=batch_size)
        self.noise_scheduler.set_timesteps(num_inference_steps = self.num_steps)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        if not self.training :
            steps = tqdm(range(self.num_steps-1,-1,-1))
        else :
            steps = range(self.num_steps-1,-1,-1)
        for step in steps:
            with torch.no_grad():
                model_output = self(xt,x,timesteps[step])
                # 2. compute previous image: x_t -> x_t-1
                xt = self.noise_scheduler.step(
                                        model_output= model_output,
                                        timestep = step,
                                        sample= xt).prev_sample
        return xt   
    
    def validation_step(self,batch, batch_idx):
        x,y = batch
        xt = self.sample(x)
        MSE = F.mse_loss(y,xt)
        self.log("MSE_loss",MSE, on_step=True, on_epoch=True, prog_bar=True,rank_zero_only=True,sync_dist=True)
        return {
            "MSE": MSE,
        }
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return [optimizer]


if __name__ == "__main__":
    model = Diffusion_UNET(in_channels=1).to("cuda")
    x = torch.randn(16, 1, 32, 32).to("cuda")
    t = torch.Tensor([16]).to("cuda")
    """
    out = model(x,x,t)
    print(out.shape)
    """
    model.training_step((x,x),0)