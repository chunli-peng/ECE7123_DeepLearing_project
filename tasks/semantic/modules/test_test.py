from convnext_model import ConvNeXt
import torch

with torch.no_grad():
	model = ConvNeXt(in_chans=5, dims=[64, 128, 256, 512])

model.train()
print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
a = torch.rand(1, 5, 64, 2048)
y = model(a)
print(y)
