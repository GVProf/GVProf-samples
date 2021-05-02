import torch
import math
import time

device = torch.device("cuda:0")  # Uncomment this to run on GPU

channels = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]

inputs = [torch.randn((1, 64, 56, 56), device=device),
          torch.randn((1, 128, 28, 28), device=device),
          torch.randn((1, 256, 14, 14), device=device),
          torch.randn((1, 512, 7, 7), device=device)]

convs = [3, 4, 6, 3]

with torch.no_grad():
    input = torch.randn((1, 3, 224, 224), device=device) 
    conv7x7 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, groups=1, bias=False, dilation=1).cuda()

    for _ in range(100):
        conv7x7(input)

    for i in range(len(channels)):
        channel = channels[i]
        conv = convs[i]
        conv3x3 = torch.nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1).cuda()
        conv1x1_1 = torch.nn.Conv2d(channel[0], channel[0], kernel_size=1, stride=1, padding=1, groups=1, bias=False, dilation=1).cuda()
        conv1x1_2 = torch.nn.Conv2d(channel[1], channel[2], kernel_size=1, stride=1, padding=1, groups=1, bias=False, dilation=1).cuda()

        input = inputs[i]

        for _ in range(100):
            for _ in range(conv):
                output = conv3x3(input)
                output = conv1x1_1(input)
                output = conv1x1_2(input)

#NOTE: use nsys to get kernel time, this script involves extra memset time
