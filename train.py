'''
The model training script.
This script uses the SIS, SIR, SGD, or Adam method to train a neural network.

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import OneLayerBinaryClassifier, SmallCNN
from particle_filter import SIS, SIR, SGD, Adam
from csv_loader import KidsHeightDataLoader


######################################################
###  Hyper Parameters  ###############################
######################################################
NUM_PARTICLES = 100
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.1
PROCESS_NOISE = 0.1
DEVICE = torch.device("cuda:0")
MODEL_SAVE_PATH = 'name.model'


########################################################
#######  Utility Functions  ############################
########################################################
def min_med_max(array):
  median = np.median(array)
  minimum = np.amin(array)
  maximum = np.amax(array)
  return minimum, median, maximum


#########################################################
#########  Load Dataset  ################################
#########################################################
print("Load dataset")
'''
pipeline = [
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ]
transform = transforms.Compose(pipeline)   
trainset = torchvision.datasets.CIFAR10(root='./', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4)
'''

trainloader = KidsHeightDataLoader('KidsHeightData.csv', \
                                   batch_size=BATCH_SIZE, shuffle=True)


##########################################################
#######  Train  ##########################################
##########################################################
print('Start Training')
trainer = SIR(OneLayerBinaryClassifier, NUM_PARTICLES, PROCESS_NOISE, \
              NUM_PARTICLES * 0.6)
#trainer = SGD(OneLayerBinaryClassifier, NUM_PARTICLES, LEARNING_RATE)

running_loss = np.zeros(NUM_PARTICLES)
         
for epoch in range(EPOCHS):
  print("\nEpoch", epoch)  
  start = time.time()
  
  for step, data in enumerate(trainloader):
    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    
    losses = trainer.train_step(images, labels)
    running_loss += np.array(losses)
    
    if step % 200 == 199:
      running_loss /= 200
      
      if NUM_PARTICLES > 4:
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
 
 
########################################################
####  Save trained model  ##############################
########################################################
print('Save model to file:', MODEL_SAVE_PATH)
min_imp, median_imp, max_imp = min_med_max(trainer.importance)
print(f"importance {min_imp:.6g} ... {median_imp:.6g} ... {max_imp:.6g}")

# Use Maximum A Posteriori (MAP) estimate on the probability distribution of
# the weights
best_particle = np.argmax(trainer.importance)

torch.save(trainer.particles.nets[best_particle].state_dict(), MODEL_SAVE_PATH)

