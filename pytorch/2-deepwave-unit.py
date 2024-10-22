import torch
import math
import time

dtype = torch.float
device = torch.device("cuda:0")  # Uncomment this to run on GPU

m = torch.nn.ReplicationPad3d(3).cuda()
input = torch.randn((1, 16, 100, 100, 100), device=device, requires_grad=True)
grad = torch.randn((1, 16, 106, 106, 106), device=device)
output = torch.randn((1, 16, 106, 106, 106), device=device)

t = time.time()
for i in range(300):
    output = m(input)
    output.backward(grad)
print(output[0][0][0][0][0])
print('time: {:.6f}s'.format((time.time() - t) / 300))
