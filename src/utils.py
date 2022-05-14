import random

import torch

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
cuda = lambda o: o.cuda() if use_cuda else o
tensor = lambda o: cuda(torch.tensor(o))
eye = lambda d: cuda(torch.eye(d))
zeros = lambda *args: cuda(torch.zeros(*args))

crandn = lambda *args: cuda(torch.randn(*args))
detach = lambda o: o.cpu().detach().numpy().tolist()


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
