import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ]
)
d_in = 3
d_out = 2
torch.manual
w_query = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
w_key = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
w_value = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)

keys = inputs @ w_key
values = inputs @ w_value
query = inputs @ w_query
print(keys)
print(values)
print(query)