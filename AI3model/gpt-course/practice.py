# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:36:44 2024

@author: user
"""
import torch
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time= time.time()

zeros = torch.zeros(1, 1)
end_time = time.time()

elapsed_time = end_time -start_time
print(f"{elapsed_time: 4f")

