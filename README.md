# particle_filter
Use a particle filter to estimate the parameters in a neural network.

Based on the paper: Freitas, J. D., Niranjan, M., Gee, A. H., & Doucet, A. (2000). Sequential Monte Carlo methods to train neural network models. Neural computation, 12(4), 955-993.

The purpose of this project is to compare particle filter to the standard back propogation algorightm in their abilities to train neural networks.

A report on the methods and findings of this project is in file [report.pdf](./report.pdf)

A video presentation of this project is on Youtube https://www.youtube.com/watch?v=9nCohdFoNCE

[![YouTube video](http://img.youtube.com/vi/9nCohdFoNCE/0.jpg)](http://www.youtube.com/watch?v=9nCohdFoNCE)

# File structure

1. train.py - the main script that trains a neural network model </br>
            └-- particle.filter.py - contains the particle filter algorithm as well asgradient descent based algorithms       
            └-- model.py - contains the definition of neural network architectures          
            └-- csv_loader.py - a small module that loads the Kids Height data set
         
2. test_model.py - runs the trained model on the test data set to check for prediction accuracy </br>
                 └-- model.py              
                 └-- csv_loader.py

3. examine_parameters.py - displays the parameters of the trained neural network

