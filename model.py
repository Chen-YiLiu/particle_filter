'''
This file defines the base neural network models.

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################
###  a single layer with two inputs  #################
######################################################
class OneLayerBinaryClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(2, 1)
    self.bceloss = nn.BCEWithLogitsLoss()
  
  def forward(self, x):
    x = self.fc1(x)
    return x
  
  def predict(self, y_est):
    prediction = torch.sigmoid(y_est)
    return prediction.round().long().squeeze()
  
  def calculate_loss(self, y_est, y):
    y = torch.unsqueeze(y, 1)
    y = y.float()
    self.loss = self.bceloss(y_est, y)    
    return self.loss.item() 


######################################################
#########  a small CNN for Cifar-10  #################
######################################################
class SmallCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # 3 x 32 x 32
    self.conv1 = nn.Conv2d(3, 10, 3)
    # 10 x 30 x 30
    self.conv1_bn = nn.BatchNorm2d(10)
    self.pool = nn.MaxPool2d(2, stride=2)
    # 10 x 15 x 15
    self.conv2 = nn.Conv2d(10, 10, 3, stride=2)
    # 10 x 7 x 7
    self.conv2_bn = nn.BatchNorm2d(10)
    self.conv3 = nn.Conv2d(10, 10, 3, stride=2)
    # 10 x 3 x 3
    self.conv3_bn = nn.BatchNorm2d(10)
    self.fc1 = nn.Linear(10 * 3 * 3, 10)
    self.celoss = nn.CrossEntropyLoss()

  def forward(self, x):    
    x = self.conv1(x)
    x = self.conv1_bn(x)
    x = F.relu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.conv2_bn(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.conv3_bn(x)
    x = F.relu(x)
    x = self.fc1(x.view(-1, 10 * 3 * 3))
    return x
    
  def calculate_loss(self, y_est, y):
    self.loss = self.celoss(y_est, y)    
    return self.loss.item()
  
  def predict(self, y_est):
    return torch.argmax(y_est, 1)


######################################################
#########  a large CNN for Cifar-10  #################
######################################################
class BigCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # 3 x 32 x 32
    self.conv1 = nn.Conv2d(3, 32, 3)
    # 32 x 30 x 30
    self.conv1_bn = nn.BatchNorm2d(32)
    self.pool = nn.MaxPool2d(2, stride=2)
    # 32 x 15 x 15
    self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
    # 64 x 7 x 7
    self.conv2_bn = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, 3)
    # 64 x 5 x 5
    self.conv3_bn = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 32, 1)
    # 32 x 5 x 5
    self.conv4_bn = nn.BatchNorm2d(32)
    self.fc1_input_channels = 32 * 5 * 5
    self.fc1 = nn.Linear(self.fc1_input_channels, 64)
    self.fc2 = nn.Linear(64, 10)
    #self.softmax = nn.Softmax(dim=1)
    #self.nllloss = nn.NLLLoss()
    #self.mseloss = nn.MSELoss()
    self.celoss = nn.CrossEntropyLoss()

  def forward(self, x):    
    x = self.conv1(x)
    x = self.conv1_bn(x)
    x = F.relu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.conv2_bn(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.conv3_bn(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = self.conv4_bn(x)
    x = F.relu(x)
    x = self.fc1(x.view(-1, self.fc1_input_channels))
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    #x = self.softmax(x)
    return x
    
  def calculate_loss(self, y_est, y):
    '''
    log_prob = torch.log(y_est)     
    self.loss = self.nllloss(log_prob, y)
    self.loss = self.mseloss(y_est, y)
    '''
    self.loss = self.celoss(y_est, y)    
    return self.loss.item()
  
  def predict(self, y_est):
    return torch.argmax(y_est, 1)
    

