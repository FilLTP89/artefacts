import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm
from copy import copy 


class ImageSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1):
        super(ImageSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.kernel_size = kernel_size

        # Define the convolution layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        # Initialize weights
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Compute attention scores
        attention = torch.bmm(query, key)
        attention = self.softmax(attention / (self.in_channels ** 0.5))

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.out_channels, height, width)
        return out
    

def same_padding(kernel_size, stride):
    pad = (kernel_size - stride + 1) // 2
    return pad
class AttentionGate(nn.Module):
    def __init__(self, 
                g_channels,
                x_channels,
                out_channels, **kwargs):
        super(AttentionGate, self).__init__()

        self.g_channels = g_channels
        self.x_channels = x_channels
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, out_channels , kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels,out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.phi = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.att = ImageSelfAttention(out_channels, out_channels)

    def forward(self, x, g):
        #print(f"g channels : {self.g_channels}, g shape : {g.shape}")
        #print(f"x channels : {self.x_channels}, x shape : {x.shape}")
        g = self.W_g(g)
        x = self.W_x(x)
        phi = self.relu(g + x)
        phi = self.phi(phi)
        phi = self.att(phi)
        return x * phi

class U_block(nn.Module):
    def __init__(self, 
                 shape=(1, 512, 512),
                 filters = [8,16,32, 64,128,256,512,1024]
                 ):
        super(U_block, self).__init__()
        self.shape = shape
        self.filters = filters
        

        self.original_conv = nn.Conv2d(self.shape[0], self.filters[0], 1, 1, padding=0)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.bottleneck = self._build_bottleneck()
        self.final_conv = nn.Conv2d(self.filters[1], 1, 1, 1, padding=0)
        
        self.attention_gates = nn.ModuleList([
            AttentionGate(g_channels=self.filters[i], x_channels=self.filters[i], out_channels=self.filters[i-1])
            for i in range(len(self.filters)-1, -1, -1)
        ])

    def _build_encoder(self):
        layers = [self.original_conv]
        in_channels = self.filters[0]
        for filter in self.filters[1:]:
            layers.append(self._down_conv_block(in_channels, filter))
            in_channels = filter
        return nn.ModuleList(layers)

    def _build_decoder(self):
        layers = []
        in_channels = self.filters[-1] * 2  # Doubled because of the bottleneck
        for filter in self.filters[::-1]:
            layers.append(self._up_conv_block(in_channels, filter))
            in_channels = filter * 2  # Doubled because of skip connection
        return nn.ModuleList(layers)

    def _build_bottleneck(self):
        return self._down_conv_block(self.filters[-1], self.filters[-1]*2)

    def _down_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, padding=same_padding(4, 2)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)

        x = self.bottleneck(encoder_outputs[-1])
        for i, (layer) in enumerate(self.decoder):
            x = layer(x)
            skip = encoder_outputs[-(i+1)]
            #x = attention(x, skip)
            if x.shape != skip.shape:
                """
                Modify this if bad results
                """
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        x = self.final_conv(x)
        return x

class ConsNet(nn.Module):
    def __init__(self, 
                 n_block=6, 
                 input_shape=(1, 512, 512),
                 filters = [8,16,32, 64,128,256,512,1024]

                 ):
        super(ConsNet, self).__init__()
        self.shape = input_shape
        self.filters = filters
        self.Ublock = nn.ModuleList([U_block(self.shape) for _ in range(n_block)])

    def forward(self, x):
        for block in self.Ublock:
            x = block(x)
        return torch.sigmoid(x)

class PatchGAN(nn.Module):
    def __init__(self, input_shape):
        super(PatchGAN, self).__init__()
        self.model = self._build_model(input_shape[0])

    def _build_model(self, in_channels):
        def conv_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        model = nn.Sequential(
            *conv_block(in_channels, 64, normalize=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        return model

    def forward(self, x):
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        return features[:-1], x  # Return features and last layer output

class VGG19(pl.LightningModule):
    def __init__(
        self,
        shape=(577, 577, 1),
        classifier_training=False,
        load_whole_architecture=False,
        n_class = 31,
        learning_rate=1e-3,
        *args, **kwargs
    ):
        super().__init__()
        self.shape = shape
        self.learning_rate = learning_rate  
        self.n_class = n_class
        self.classifier_training = classifier_training
        self.load_whole_architecture = load_whole_architecture
        
        print(f"n_class : {self.n_class}")
        # block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 5
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        
        # Only need the last part in case we are training vgg as a classifier
        if self.classifier_training or self.load_whole_architecture:
            self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flat = nn.Flatten()
            self.dense1 = nn.Linear(578 * 16 * 16, 1024)  # Adjust input size based on your needs
            self.dense2 = nn.Linear(1024, 1024)
            self.classifier = nn.Linear(1024, n_class)
            self.loss = nn.CrossEntropyLoss() # nn.LogSoftmax() and nn.NLLLoss()
            self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_class)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.maxpool1(x1)
        x2 = self.relu2(self.conv2(x2))
        x3 = self.maxpool2(x2)
        x3 = self.relu3(self.conv3(x3))
        x4 = self.relu4(self.conv4(x3))
        x5 = self.maxpool3(x4)
        x5 = self.relu5(self.conv5(x5))
        x6 = self.relu6(self.conv6(x5))
        x7 = self.maxpool4(x6)
        x7 = self.relu7(self.conv7(x7))
        x8 = self.relu8(self.conv8(x7))
        
        if self.classifier_training:
            x = self.maxpool5(x8)
            x = self.flat(x)
            x = torch.relu(self.dense1(x))
            x = torch.relu(self.dense2(x))
            x = self.classifier(x)
            return x
        
        return [x1, x2, x3, x4, x5, x6, x7, x8]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        pred = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(pred, y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(pred, y)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True,sync_dist=True)
        return acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

class AttentionMEDGAN(pl.LightningModule):
    def __init__(
        self,
        input_shape=(1, 512, 512),
        generator=None,
        discriminator=None,
        feature_extractor=None,
        learning_rate=3e-5,
        N_g=3,
        vgg_whole_arc=False,
        cosine_decay=True,
        filters =  [8,16,32, 64,128,256,512,1024]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.shape = input_shape
        self.N_g = N_g
        self.lambda_1 = 20
        self.lambda_2 = 1e-4
        self.lambda_3 = 1
        self.lambda_4 = 1
        self.automatic_optimization = False  # This is crucial for manual optimization
        self.learning_rate = learning_rate  
        self.cosine_decay = cosine_decay

        self.generator = generator or ConsNet(3, self.shape, filters=filters)
        self.discriminator = discriminator or PatchGAN(self.shape)
        if feature_extractor :
            self.feature_extractor = feature_extractor 
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
            pass
        else :
            self.feature_extractor = VGG19(self.shape, load_whole_architecture=vgg_whole_arc)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        if cosine_decay:
            self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=1000, eta_min=0)
            self.d_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.d_optimizer, T_max=1000, eta_min=0)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        
        if self.cosine_decay:
            g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=1000, eta_min=0)
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=1000, eta_min=0)
            return [g_opt, d_opt], [g_scheduler, d_scheduler]
        else:
            return [g_opt, d_opt]


    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        real_x, real_y = batch
        
        # Get optimizers
        g_opt, d_opt = self.optimizers()

        # Train Generator
        for _ in range(self.N_g):
            g_opt.zero_grad()
            fake_y = self.generator(real_x)
            fake_features, fake_output = self.discriminator(fake_y)
            real_features, _ = self.discriminator(real_y)

            fake_vgg_features = self.feature_extractor(fake_y)
            real_vgg_features = self.feature_extractor(real_y)

            g_loss = self.generator_loss(fake_output, real_features, fake_features, real_vgg_features, fake_vgg_features, real_y, fake_y)
            self.manual_backward(g_loss)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            g_opt.step()

        # Train Discriminator
        d_opt.zero_grad()
        fake_y = self.generator(real_x)  # Generate new fakes for discriminator training
        _, real_output = self.discriminator(real_y)
        _, fake_output = self.discriminator(fake_y.detach())
        d_loss = self.discriminator_loss(real_output, fake_output)
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        d_opt.step()

        # Log losses
        self.log('g_loss', g_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('d_loss', d_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('perceptual_loss', self.perceptual_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('style_loss', self.style_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('content_loss', self.content_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('mse_loss', self.mse_loss, prog_bar=True,  sync_dist=True, rank_zero_only=True)
        self.log('real_loss', self.real_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('fake_loss', self.fake_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)   
        
        return {'g_loss': g_loss,
                'd_loss': d_loss,
                'perceptual_loss': self.perceptual_loss,
                'style_loss': self.style_loss,
                'content_loss': self.content_loss,
                'mse_loss': self.mse_loss,
                'real_loss': self.real_loss,
                'fake_loss': self.fake_loss}

    
    def test_training_step(self, batch, batch_idx):
        real_x, real_y = batch
        
        # Get optimizers
        g_opt, d_opt = self.g_optimizer, self.d_optimizer
        # Train Generator
        for _ in range(self.N_g):
            g_opt.zero_grad()
            fake_y = self.generator(real_x)
            fake_features, fake_output = self.discriminator(fake_y)
            real_features, _ = self.discriminator(real_y)

            fake_vgg_features = self.feature_extractor(fake_y)
            real_vgg_features = self.feature_extractor(real_y)

            g_loss = self.generator_loss(fake_output, real_features, fake_features, real_vgg_features, fake_vgg_features, real_y, fake_y)

        # Train Discriminator
        d_opt.zero_grad()
        fake_y = self.generator(real_x)  # Generate new fakes for discriminator training
        _, real_output = self.discriminator(real_y)
        _, fake_output = self.discriminator(fake_y.detach())
        d_loss = self.discriminator_loss(real_output, fake_output)

        return {'g_loss': g_loss,
                'd_loss': d_loss,
                'perceptual_loss': self.perceptual_loss,
                'style_loss': self.style_loss,
                'content_loss': self.content_loss,
                'mse_loss': self.mse_loss,
                'real_loss': self.real_loss,
                'fake_loss': self.fake_loss}
    

    def validation_step(self, batch, batch_idx):
        real_x, real_y = batch
        for _ in range(self.N_g):
            fake_y = self.generator(real_x)
            fake_features, fake_output = self.discriminator(fake_y)
            real_features, _ = self.discriminator(real_y)

            fake_vgg_features = self.feature_extractor(fake_y)
            real_vgg_features = self.feature_extractor(real_y)

            g_loss = self.generator_loss(fake_output, real_features, fake_features, real_vgg_features, fake_vgg_features, real_y, fake_y)

        # Train Discriminator
        fake_y = self.generator(real_x)  # Generate new fakes for discriminator training
        _, real_output = self.discriminator(real_y)
        _, fake_output = self.discriminator(fake_y.detach())
        d_loss = self.discriminator_loss(real_output, fake_output)

        self.log('test_g_loss', g_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_d_loss', d_loss, prog_bar=True, sync_dist=True , rank_zero_only=True)
        self.log('test_perceptual_loss', self.perceptual_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_style_loss', self.style_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_content_loss', self.content_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_mse_loss', self.mse_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_real_loss', self.real_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('test_fake_loss', self.fake_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)


        return {'g_loss': g_loss, 
                'd_loss': d_loss, 
                'perceptual_loss': self.perceptual_loss,
                'style_loss': self.style_loss,
                'content_loss': self.content_loss,
                'mse_loss': self.mse_loss,
                'real_loss': self.real_loss,
                'fake_loss': self.fake_loss
                }

    def on_train_epoch_end(self):
        # Step the learning rate schedulers if they exist
        if self.cosine_decay:
            sch = self.lr_schedulers()
            sch[0].step()  # generator scheduler
            sch[1].step()  # discriminator scheduler

    def generator_loss(self, fake_output, real_features, fake_features, real_vgg_features, fake_vgg_features, real_y, fake_y):
        gan_loss = nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))
        perceptual_loss = nn.MSELoss()(fake_features[-1], real_features[-1])
        style_loss = self.compute_style_loss(fake_vgg_features, real_vgg_features)
        
        content_loss = self.compute_content_loss(fake_vgg_features, real_vgg_features)
        mse_loss = nn.MSELoss()(fake_y, real_y)

        self.perceptual_loss = perceptual_loss
        self.style_loss = style_loss
        self.content_loss = content_loss
        self.mse_loss = mse_loss

        return (gan_loss + 
                self.lambda_1 * perceptual_loss + 
                self.lambda_2 * style_loss + 
                self.lambda_3 * content_loss + 
                self.lambda_4 * mse_loss)
    
    def compute_content_loss(self, fake_features, real_features):
        content_loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            content_loss += nn.MSELoss()(fake_feat, real_feat)
        return content_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
        fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))

        self.real_loss = real_loss
        self.fake_loss = fake_loss

        return (real_loss + fake_loss) / 2

    
    def compute_style_loss(self, fake_features, real_features):
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)

        style_loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            style_loss += nn.MSELoss()(gram_matrix(fake_feat), gram_matrix(real_feat))
        return style_loss


class OptimizedAttentionMEDGAN(pl.LightningModule):
    def __init__(
        self,
        input_shape=(1, 512, 512),
        generator=None,
        discriminator=None,
        feature_extractor=None,
        g_lr=2e-4,
        d_lr=4e-4,
        N_g=3,
        vgg_whole_arc=False,
        cosine_decay=True,
        filters=[8,16,32,64,128,256,512,1024],
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.shape = input_shape
        self.N_g = N_g
        
        # Adjusted loss weights based on typical GAN implementations
        self.lambda_1 = 10.0  # Perceptual loss weight (reduced from 20)
        self.lambda_2 = 1.0   # Style loss weight (increased from 1e-4)
        self.lambda_3 = 1.0   # Content loss weight
        self.lambda_4 = 10.0  # MSE loss weight (increased for better reconstruction)
        self.lambda_gp = 10.0 # Gradient penalty weight
        
        self.automatic_optimization = False
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.cosine_decay = cosine_decay

        # Networks (keeping your original architectures)
        self.generator = generator or ConsNet(6, self.shape, filters=filters)
        self.discriminator = discriminator or PatchGAN(self.shape)
        
        if feature_extractor:
            self.feature_extractor = feature_extractor
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = VGG19(self.shape, load_whole_architecture=vgg_whole_arc)
            
        # Historical averaging for generator
        self.register_buffer('generator_ema', None)
        self.ema_beta = 0.999
        
        # Initialize metrics
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        
        if self.cosine_decay:
            g_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                g_opt,
                max_lr=self.g_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2
            )
            d_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                d_opt,
                max_lr=self.d_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2
            )
            return [g_opt, d_opt], [g_scheduler, d_scheduler]
        return [g_opt, d_opt]

    def compute_gradient_penalty(self, real_y, fake_y):
        """Compute gradient penalty for WGAN-GP"""
        alpha = torch.rand((real_y.size(0), 1, 1, 1), device=self.device)
        interpolated = (alpha * real_y + (1 - alpha) * fake_y).requires_grad_(True)
        
        _, d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        real_x, real_y = batch
        
        # Get optimizers
        g_opt, d_opt = self.optimizers()

        # Train Generator
        for _ in range(self.N_g):
            g_opt.zero_grad()
            fake_y = self.generator(real_x)
            fake_features, fake_output = self.discriminator(fake_y)
            real_features, _ = self.discriminator(real_y)

            fake_vgg_features = self.feature_extractor(fake_y)
            real_vgg_features = self.feature_extractor(real_y)

            g_loss = self.generator_loss(fake_output, real_features, fake_features, 
                                       real_vgg_features, fake_vgg_features, real_y, fake_y)
            self.manual_backward(g_loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            g_opt.step()

        # Train Discriminator
        d_opt.zero_grad()
        fake_y = self.generator(real_x)
        _, real_output = self.discriminator(real_y)
        _, fake_output = self.discriminator(fake_y.detach())
        
        # Compute gradient penalty
        grad_penalty = self.compute_gradient_penalty(real_y, fake_y)
        
        d_loss = self.discriminator_loss(real_output, fake_output)
        d_loss = d_loss + self.lambda_gp * grad_penalty
        
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        d_opt.step()

        # Update EMA generator
        if self.generator_ema is None:
            self.generator_ema = copy.deepcopy(self.generator.state_dict())
        else:
            self.update_ema_generator()

        # Logging
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'grad_penalty': grad_penalty,
            'perceptual_loss': self.perceptual_loss,
            'style_loss': self.style_loss,
            'content_loss': self.content_loss,
            'mse_loss': self.mse_loss,
            'real_loss': self.real_loss,
            'fake_loss': self.fake_loss
        }, prog_bar=True, sync_dist=True)
        
        return {'g_loss': g_loss, 'd_loss': d_loss}

    def update_ema_generator(self):
        with torch.no_grad():
            for param, ema_param in zip(self.generator.state_dict().items(), 
                                      self.generator_ema.items()):
                ema_param[1].data.mul_(self.ema_beta).add_(
                    param[1].data, alpha=1 - self.ema_beta)

    def generator_loss(self, fake_output, real_features, fake_features, 
                      real_vgg_features, fake_vgg_features, real_y, fake_y):
        # Adversarial loss (using hinge loss)
        gen_loss = -fake_output.mean()
        
        # Feature matching loss
        feat_matching_loss = 0
        for real_feat, fake_feat in zip(real_features[:-1], fake_features[:-1]):
            feat_matching_loss += F.l1_loss(fake_feat, real_feat)

        perceptual_loss = nn.MSELoss()(fake_features[-1], real_features[-1])
        style_loss = self.compute_style_loss(fake_vgg_features, real_vgg_features)
        content_loss = self.compute_content_loss(fake_vgg_features, real_vgg_features)
        mse_loss = nn.MSELoss()(fake_y, real_y)

        self.perceptual_loss = perceptual_loss
        self.style_loss = style_loss
        self.content_loss = content_loss
        self.mse_loss = mse_loss

        return (gen_loss + 
                self.lambda_1 * perceptual_loss + 
                self.lambda_2 * style_loss + 
                self.lambda_3 * content_loss + 
                self.lambda_4 * mse_loss + 
                0.1 * feat_matching_loss)  # Small weight for feature matching

    def discriminator_loss(self, real_output, fake_output):
        # Hinge loss for discriminator
        self.real_loss = torch.mean(F.relu(1.0 - real_output))
        self.fake_loss = torch.mean(F.relu(1.0 + fake_output))
        return self.real_loss + self.fake_loss

    def compute_style_loss(self, fake_features, real_features):
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)

        style_loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            style_loss += nn.MSELoss()(gram_matrix(fake_feat), gram_matrix(real_feat))
        return style_loss

    def compute_content_loss(self, fake_features, real_features):
        content_loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            content_loss += nn.MSELoss()(fake_feat, real_feat)
        return content_loss

    def validation_step(self, batch, batch_idx):
        real_x, real_y = batch
        
        # Use EMA generator for validation
        if self.generator_ema is not None:
            generator_state = self.generator.state_dict()
            self.generator.load_state_dict(self.generator_ema)
        
        with torch.no_grad():
            fake_y = self.generator(real_x)
            fake_features, fake_output = self.discriminator(fake_y)
            real_features, real_output = self.discriminator(real_y)

            fake_vgg_features = self.feature_extractor(fake_y)
            real_vgg_features = self.feature_extractor(real_y)

            g_loss = self.generator_loss(fake_output, real_features, fake_features,
                                       real_vgg_features, fake_vgg_features, real_y, fake_y)
            d_loss = self.discriminator_loss(real_output, fake_output)
        
        # Restore generator weights if using EMA
        if self.generator_ema is not None:
            self.generator.load_state_dict(generator_state)

        self.log_dict({
            'val_g_loss': g_loss,
            'val_d_loss': d_loss,
            'val_perceptual_loss': self.perceptual_loss,
            'val_style_loss': self.style_loss,
            'val_content_loss': self.content_loss,
            'val_mse_loss': self.mse_loss
        }, prog_bar=True, sync_dist=True)

        return {'val_g_loss': g_loss, 'val_d_loss': d_loss}

    def on_train_epoch_end(self):
        if self.cosine_decay:
            sch = self.lr_schedulers()
            sch[0].step()
            sch[1].step()
    
if __name__ == "__main__":
    input_shape = (1, 577, 577)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMEDGAN(
        input_shape=(1, 577, 577)
    ).to(device)

    x = torch.randn(1, 1, 577,577).to(device)
    y = torch.randn(1, 1, 577,577).to(device)
    loss = model.test_training_step((x, y), 0)   
    print(loss)
    summary(model, (1, 577,577))
    """
    vgg = VGG19(classifier_training=True, n_class=31).to("cuda")
    x = torch.randn((2,1,557,557)).to("cuda")
    y = torch.ones([2]).type(torch.LongTensor).to("cuda")
    print(y.shape)
    pred = vgg(x)
    loss = nn.CrossEntropyLoss()
    v_loss = loss(pred,y)
    print(v_loss)
    """