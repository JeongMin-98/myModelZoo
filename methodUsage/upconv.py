from torch import nn
import torch

ipt = torch.randn(1, 4, 4)

print(ipt.shape)

conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
upconv = nn.ConvTranspose2d(1,1, kernel_size=3)

result = conv(ipt)

print(result.shape)

upsample = upconv(result)
print(upsample.shape)