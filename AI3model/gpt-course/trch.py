import torch

randint=torch.randint(-100,100,(6,))
tensor=torch.tensor([[0.1,1.2],[2.2,3.1],[4.9,5.2]])
zeros=torch.zeros(2,3)
ones=torch.ones(3,4)
empty=torch.empty(2,3)
arange=torch.arange(5)
linespace =torch.linspace(3,10,steps=5)
logspace=torch.logspace(-10,10,steps=5)
eye=torch.eye(5)
music=torch.randn(5,5,(2))
a = torch.empty((2,3), dtype=torch.int64)
empty_like =torch.empty_like(a)


