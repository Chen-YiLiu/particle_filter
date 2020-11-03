'''
Try different particle sizes to see how it affects the required training steps

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import time
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import OneLayerBinaryClassifier
from particle_filter import SIS, SIR, SGD, Adam
from csv_loader import KidsHeightDataLoader


######################################################
###  Hyper Parameters  ###############################
######################################################
NUM_PARTICLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BATCH_SIZE = 1
MAX_EPOCHS = 100000
LEARNING_RATE = 0.1
PROCESS_NOISE = 0.1
STOPPING_CRITERIUM = 0.9
DEVICE = torch.device("cuda:0")
RESULT_SAVE_PATH = 'steps_results.txt'


########################################################
#######  Utility Functions  ############################
########################################################
def min_med_max(array):
  median = np.median(array)
  minimum = np.amin(array)
  maximum = np.amax(array)
  return minimum, median, maximum


def evaluate(net):
  total_loss = 0
  total_correct = 0
  with torch.no_grad(): 
    for data in testloader:
      images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    
      output = net(images)
     
      loss = net.calculate_loss(output, labels)
      total_loss += loss
    
      predictions = net.predict(output)
      total_correct += torch.sum(predictions == labels).item()
      
  avg_loss = total_loss * 100 / len(testloader)
  accuracy = total_correct / len(testloader)  
  return avg_loss, accuracy
    

#########################################################
#########  Load Dataset  ################################
#########################################################
print("Load dataset")
trainloader = KidsHeightDataLoader('KidsHeightData.csv', \
                                   batch_size=BATCH_SIZE, shuffle=True)
testloader = KidsHeightDataLoader('KidsHeightData.csv', batch_size=100)


##########################################################
#######  Train  ##########################################
##########################################################
print('Start Training')
steps_required = []

for num_particles in NUM_PARTICLES:
  print(f"\nUse {num_particles} particles")
  #trainer = SGD(OneLayerBinaryClassifier, num_particles, LEARNING_RATE)
  trainer = SIR(OneLayerBinaryClassifier, num_particles, PROCESS_NOISE, \
                num_particles * 0.6)
  
  # initialize parameters to range uniformly from -100 to 100
  parameters = trainer.particles.state_dict()
  new_parameters = {}
  for name, tensor in parameters.items():
    shape = tensor.size()
    initial = np.random.rand(*shape) * 200 -100
    new_parameters[name] = torch.from_numpy(initial)
  trainer.particles.load_state_dict(new_parameters)
  
  running_loss = np.zeros(num_particles)
         
  for epoch in range(MAX_EPOCHS):
    print("\nEpoch", epoch)  
    start = time.time()
  
    for step, data in enumerate(trainloader):
      images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    
      losses = trainer.train_step(images, labels)
      running_loss += np.array(losses)
    
      if step % 200 == 199:
        running_loss /= 200
      
        if num_particles > 4:
          min_loss, median_loss, max_loss = min_med_max(running_loss)
          min_imp, median_imp, max_imp = min_med_max(trainer.importance)
          loss_str = f"loss {min_loss:.3g} ~ {median_loss:.3g} ~ {max_loss:.3g}"
          imp_str = f"importance {min_imp:.3g} ~ {median_imp:.3g} ~ {max_imp:.3g}"
          print(f"step {step}", loss_str, imp_str)
      
        else:
          print(f"step {step} \tloss {running_loss}")
          #print("\timportance", trainer.importance)
          
        running_loss *= 0

    end = time.time()
    print("Epoch training time (s):", end - start)
    
    # use the particle with highest importance ratio to check for performance
    # terminate training if accuracy or loss meets the stopping criteria
    best_particle = np.argmax(trainer.importance)
    avg_loss, accuracy = evaluate(trainer.particles.nets[best_particle])
    print(f"avg. loss {avg_loss:.3g},\t accuracy {accuracy:.3g}")
    if accuracy >= STOPPING_CRITERIUM:
      steps_required.append([num_particles, epoch + 1])
      break
    elif epoch == MAX_EPOCHS - 1:
      steps_required.append([num_particles, "failed"])
 
 
########################################################
####  Save steps result  ###############################
########################################################
print('Save result to file:', RESULT_SAVE_PATH)
with open(RESULT_SAVE_PATH, 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerows(steps_required)
    
