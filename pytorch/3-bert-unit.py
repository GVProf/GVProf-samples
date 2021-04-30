import torch
import math
import time

dtype = torch.float
device = torch.device("cuda:0")  # Uncomment this to run on GPU

m = torch.nn.Embedding(200, 512, padding_idx=0).cuda()
input = torch.randint(low=0, high=199, size=(50, 20), device=device)

t = time.time()
for i in range(20000):
    output = m(input)
print('time: {:.4f}s'.format(time.time() - t))

