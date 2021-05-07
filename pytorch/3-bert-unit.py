import torch
import math
import time

dtype = torch.float
device = torch.device("cuda:0")  # Uncomment this to run on GPU

m = torch.nn.Embedding(200, 512, padding_idx=0).cuda()
input = torch.randint(low=0, high=199, size=(50, 20), device=device)

t = time.time()
for i in range(201):
    output = m(input)
#sync
print(output[0][0][0])
print('time: {:.6f}s'.format(time.time() - t))

