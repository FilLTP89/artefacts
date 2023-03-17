import torch
import torch.nn as nn

import segmentation_models_3D as sm


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = sm.Unet(
            "resnet50",
            input_shape=(32, 588, 512, 512),
            classes=1,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Model()
    x = torch.rand(32, 588, 512, 512)
    y = model(x)
    print(y.shape)

