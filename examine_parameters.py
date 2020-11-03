'''
A short script to look at the trained parameters

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import torch
import model

MODEL_PATH = 'name.model'

net = model.OneLayerBinaryClassifier()
net.load_state_dict(torch.load(MODEL_PATH))

print("model:", MODEL_PATH)
for name, value in net.named_parameters():
  print(name)
  print(value)

