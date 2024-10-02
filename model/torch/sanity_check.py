import torch
from model.torch.DIffusion_UNET import ImageToImageDDIMLightningModule



def sanity_check(model, dataloader, criterion, optimizer, device):
    model.to(device)
    model.train()

    try:
        # 1. Data Loading
        batch = next(iter(dataloader))
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # 2 & 3. Input Shape Verification and Forward Pass
        outputs = model(inputs)

        # 4. Output Shape Verification
        assert outputs.shape[0] == inputs.shape[0], "Output batch size mismatch"
        assert outputs.shape[1] == model.output_dim, "Output dimension mismatch"

        # 5. Loss Calculation
        loss = criterion(outputs, targets)
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"

        # 6. Backward Pass
        loss.backward()

        # Check if gradients are flowing
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        # 7. Optimizer Step
        optimizer.step()
        optimizer.zero_grad()

        # 8. Metric Calculation (example: accuracy)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == targets).float().mean()

        print("Sanity check passed successfully!")
        print(f"Sample loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    except Exception as e:
        print(f"Sanity check failed: {str(e)}")


if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    model = ImageToImageDDIMLightningModule()

    x_batch = torch.randn(4, 1, 512, 512)
    y_batch = torch.randn(4, 1, 512, 512)
    dataloader = [(x_batch, y_batch)]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())