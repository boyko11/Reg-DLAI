# ANN Regularization
# Week 1 Assignment 2 
# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization   

#### Visualize Data
python vizualize.py  

![Visualize Data](plots/vis_data.png)

###### The ANN will try to classify blue vs red

 
### No Regularization 

python Runner.py 0 1   

The first arg is the value of 'lambda' the regularization param  
The second is the dropout keep probability.
Both should be between 0 and 1, though no validation has been implemented.  
0 and 1 meeans, lambda is 0 => no regularization,  
the keep probability is 1 => no dropout since all the nodes are kept during training. 

Train/Test accuracy: 0.9479 / 0.915  

##### Learning Curve  
![Learning Curve](plots/learning_curve_no_reg.png)  

##### Decision Boundary
![Learning Curve](plots/dec_bound_no_reg.png)  


### Regularization lambda=0.7  
python Runner.py 0.7 1  

Train/Test accuracy: 0.938 / 0.93  

![Learning Curve](plots/learning_curve_reg.png)  

##### Decision Boundary
![Learning Curve](plots/dec_bound_reg.png)  


### Dropout
python Runner.py 0 0.86  

Train/Test accuracy: 0.929 / 0.95  

![Learning Curve](plots/learning_curve_dropout.png)  

##### Decision Boundary
![Decision Boundary](plots/dec_bound_dropout.png)  
