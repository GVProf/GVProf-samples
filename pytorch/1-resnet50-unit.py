import torch
import math
import time

device = torch.device("cuda:0")  # Uncomment this to run on GPU

conv3x3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1).cuda()
conv1x1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1).cuda()

input = torch.randn((1, 64, 56, 56), device=device)

t = time.time()
for i in range(200000):
    with torch.no_grad():
        output = conv1x1(input)
# force a sync
print(output[0][0][0][0])
print('conv1x1 time: {:.4f}s'.format(time.time() - t))

t = time.time()
for i in range(200000):
    with torch.no_grad():
        output = conv3x3(input)
# force a sync
print(output[0][0][0][0])
print('conv1x1 time: {:.4f}s'.format(time.time() - t))

