'''
Particle filter based neural network optimizer
Based on the paper: Freitas, J. D., Niranjan, M., Gee, A. H., & Doucet, A. (2000). Sequential Monte Carlo methods to train neural network models. Neural computation, 12(4), 955-993.

Two algorithms from the paper and SGD, Adam algorithms are implemented
Sequential Importance Sampling (SIS)
Sequential Importance Sampling with Resampling (SIR)
Stochastic Gradient Descent (SGD)
Adam

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda:0")


###########################################################
######  Particles  ########################################
###########################################################

class Particles(nn.Module):
  def __init__(self, network, num_particles):
    super().__init__()
    self.nets = nn.ModuleList([network() for i in range(num_particles)])
    self.num_particles = num_particles
    
  def forward(self, x): 
    return [self.nets[i](x) for i in range(self.num_particles)]
  
  def calculate_loss(self, y_est, y):    
    return [self.nets[i].calculate_loss(y_est[i], y) \
            for i in range(self.num_particles)]    
  
  def backward(self):
    for i in range(self.num_particles):
      self.nets[i].loss.backward()

     
#################################################################
#####  Sequential Importance Sampling  ##########################
#################################################################
class SIS():
  def __init__(self, network, num_particles, process_noise_mag):
    self.num_particles = num_particles
    self.noise_mag = process_noise_mag
    self.particles = Particles(network, num_particles)
    self.particles.to(DEVICE)
    self.importance = np.ones((num_particles,)) / num_particles
  
  def sample(self, input_data, labels):   
    with torch.no_grad(): 
      # add process noise to the weights to simulate state transition
      for tensor in self.particles.parameters():
        shape = tensor.size()
        noise_d = np.random.randn(*shape) * self.noise_mag
        tensor += torch.tensor(noise_d, device=DEVICE)
    
      # calculate loss of each particle
      output = self.particles(input_data)
      losses = self.particles.calculate_loss(output, labels)
    
    return losses
  
  def update_importance(self, losses):
    new_importance = np.exp(-np.array(losses))
    new_importance /= np.sum(new_importance)
    self.importance *= new_importance
    self.importance /= np.sum(self.importance)
    
  def train_step(self, input_data, labels):
    losses = self.sample(input_data, labels)
    self.update_importance(losses)
    return losses
  
    
#################################################################
#####  Sequential Importance Sampling with Resampling  ##########
#################################################################
class SIR(SIS):
  def __init__(self, network, num_particles, process_noise_mag, resample_threshold):
    super().__init__(network, num_particles, process_noise_mag)
    self.threshold = resample_threshold
    self.network = network
      
  def resample(self):
    effective_sample_size = 1 / np.sum(self.importance ** 2)
    
    if effective_sample_size < self.threshold:
      new_particles = Particles(self.network, self.num_particles)
      new_particles.to(DEVICE)
      
      # randomly select particles based on their importance
      samples = np.random.choice(self.particles.nets, self.num_particles, \
                                 p=self.importance)
      
      # transfer weights from old to new particles
      for i in range(self.num_particles):
        new_particles.nets[i].load_state_dict(samples[i].state_dict())     
      self.particles = new_particles
      
      # reset importance ratio
      self.importance = np.ones((self.num_particles,)) / self.num_particles
  
  def train_step(self, input_data, labels):
    losses = super().train_step(input_data, labels)
    self.resample()
    return losses


#################################################################
#####  Stochastic Gradient Descent  #############################
#################################################################
class SGD(SIS):
  def __init__(self, network, num_particles, learning_rate, importance_decay=0.9):
    super().__init__(network, num_particles, process_noise_mag=0)
    self.decay = importance_decay
    self.optimizer = optim.SGD(self.particles.parameters(), lr=learning_rate)
    
  def sample(self, input_data, labels):
    output = self.particles(input_data)
    losses = self.particles.calculate_loss(output, labels)
    
    self.optimizer.zero_grad()
    self.particles.backward()
    self.optimizer.step()
    
    return losses
  
  def update_importance(self, losses):
    new_importance = np.exp(-np.array(losses))
    new_importance /= np.sum(new_importance)
    self.importance = self.decay * self.importance \
                      + (1.0 - self.decay) * new_importance
    self.importance /= np.sum(self.importance)


#################################################################
#####  Adam: Adaptive Moment Estimation  ########################
#################################################################
class Adam(SGD):
  def __init__(self, network, num_particles, learning_rate):
    super().__init__(network, num_particles, learning_rate)
    self.optimizer = optim.Adam(self.particles.parameters(), lr=learning_rate)
    
    

