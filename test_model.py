'''
The model testing script.
This script evaluates the trained neural network using the test dataset.

Version: 0.3
Authour: Chen-Yi Liu
Date: May 3, 2020
'''

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import model
from csv_loader import KidsHeightDataLoader


######################################################
###  Hyper Parameters  ###############################
######################################################
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 100
MODEL_PATH = 'name.model'
NETWORK = model.OneLayerBinaryClassifier
#NETWORK = model.SmallCNN
NUM_CLASSES = 2

#########################################################
#########  Load Dataset  ################################
#########################################################
print("Load dataset")
'''
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
    
testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          num_workers=4)
'''

trainloader = KidsHeightDataLoader('KidsHeightData.csv', batch_size=BATCH_SIZE)
testset = trainloader.data

##########################################################
######  Evaluate  ########################################
##########################################################
print('Start Evaluating')

net = NETWORK()
net.load_state_dict(torch.load(MODEL_PATH))
net.to(DEVICE)

confusion = torch.zeros([NUM_CLASSES, NUM_CLASSES], dtype=torch.int32)
total_loss = 0
total_correct = 0 
       
start = time.time()

with torch.no_grad(): 
  for step, data in enumerate(trainloader):
    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    
    output = net(images)
     
    loss = net.calculate_loss(output, labels)
    total_loss += loss
    
    predictions = net.predict(output)
    total_correct += torch.sum(predictions == labels).item()
    
    for row, column in zip(labels, predictions):
      confusion[row, column] += 1
      
end = time.time()

print("average loss", total_loss * BATCH_SIZE / len(testset))
print("accuracy", total_correct / len(testset))
print("Evaluation time (s):", end - start)
confusion = confusion.numpy()
print(confusion)


with open("confusion.csv", "w") as f:
  for i in range(10):
    for j in range(10):
      f.write(str(confusion[i, j]))
      f.write(",")
    f.write("\n")

