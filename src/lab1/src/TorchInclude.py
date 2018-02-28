import torch as torch__

torch = torch__
cuda_enabled = torch.cuda.is_available()
dtype = None
FloatTensor = None
Variable = torch.autograd.Variable

if cuda_enabled:
    FloatTensor = dtype = torch.cuda.FloatTensor
else:
    FloatTensor = dtype = torch.FloatTensor
