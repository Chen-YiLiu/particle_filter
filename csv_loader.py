'''
A module to load the CSV file containing the kids height dataset

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import csv
import numpy as np
import torch


class KidsHeightDataLoader():
  def __init__(self, file_path, batch_size, shuffle=False):
    with open(file_path, newline='') as f:
      reader = csv.reader(f)
      data = [[float(x) for x in row] for row in reader]
      
    self.data = np.array(data, dtype=np.float32)
    self.batch_size = batch_size
    self.shuffle = shuffle
  
  def __iter__(self):
    if self.shuffle:
      np.random.shuffle(self.data)
    self.index = 0
    return self
  
  def __next__(self):
    if self.index + self.batch_size <= self.data.shape[0]:
      i = self.index
      j = self.index + self.batch_size
      self.index += self.batch_size
      input_data = torch.tensor(self.data[i:j, 0:2])
      labels = torch.tensor(self.data[i:j, 2], dtype=torch.int64)
      return (input_data, labels)
    else:
      raise StopIteration

  def __len__(self):
    return self.data.shape[0]
