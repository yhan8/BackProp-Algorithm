import numpy as np
import matplotlib.pyplot as plt
import copy

#Define sigmoid activation function
def sigmoid(x, derivative=False):
        #activation function and derivative
        #x: input
        #derivative: boolean. If True will return the derivative
        f = 1 / (1 + np.exp(-x))
        #derivative
        if derivative == True:
            ds = (f * (1 - f))
            return ds
        return f  
        
#Inputs and targeted outputs
inputs=[[-0.2,0.1,0.3,-0.4,1],[0.6,-0.1,0.7,-0.5,1],[0.8,0.1,-0.6,0,1]]
targets =[[0.4,0.6,0.5,0.7],[0.7,0.1,0.2,0.1],[0.1,0.3,0.2,0.9]]

#define weights of the input layer, 4 inputs + bias
weight_inputs=np.random.uniform(low=-0.5, high=0.5, size=(5,5) )

#define weights of the hidden layer, 5 hidden neurons + bias
weight_output=np.random.uniform(low=-0.5, high=0.5, size=(6,4) )

### start training ###

loss = []
for epoch in range(500):
    
    predictions = []
    
    for i in range(0,len(inputs)):
        
        ### Feed Forward ###
        
        #weight sum from inputs layer
        hidden_inputs=np.dot(inputs[i],weight_inputs)
        #activation in the hidden layer
        hidden_sig=sigmoid(hidden_inputs)
        #add bias in the hidden layer
        hidden_sig=np.append(hidden_sig,1)
    
        #weight sum from hidden layer
        output=np.dot(hidden_sig,weight_output)
        #activtation in the output layer
        predicted=sigmoid(output)
        #store predicted output
        predictions.append(predicted)
        
        ### Back Propogation ###
        
        #calculate delta from output layer to hidden layer; lr=0.5
        error = targets[i]-predicted
        derivative_output = sigmoid(output,derivative=(True))
        deltaK = error * derivative_output
        #bias added
        delta_weight_output=0.5*(np.dot(hidden_sig.reshape((6,1)),deltaK.reshape((1,4))))
        #add momentum; mc=0.9
        if epoch>1:
            delta_weight_output +=0.9 * delta_weight_output_old
        #store delta weight from previous iteration
        delta_weight_output_old = copy.deepcopy(delta_weight_output)
    
    
        #calculate delta from hidden layer to input layer
        errorJ=np.dot(weight_output,deltaK.reshape(4,1))
        derivative_outputJ=sigmoid(hidden_inputs,derivative=(True))
        #exclude bias
        errorJ=errorJ[:-1]
        deltaJ= np.asarray(errorJ) * np.reshape(derivative_outputJ,(5,1))
        inputs_array = np.asarray(inputs[i])
        #add momentum
        delta_weight=0.5*(np.dot(np.reshape(inputs_array, (5,1)),deltaJ.transpose()))
        if epoch>1:
            delta_weight +=0.9 * delta_weight_old
        delta_weight_old = copy.deepcopy(delta_weight)    
    
    
        #update weight for the hidden layer
        weight_output += delta_weight_output
        #update weight for the input layer
        weight_inputs += delta_weight
    
    ### calculate RMS ###
    predictions=np.asarray(predictions)    
    sum_error = sum(sum((targets-predictions)**2))
    rms = sum_error /(len(targets) * len(targets[0]))
    loss.append(rms)
    
  
### Feed Forward Testing ###
hidden_inputs=np.dot(inputs[0],weight_inputs)
#activation in the hidden layer
hidden_sig=sigmoid(hidden_inputs)
#weight sum, add bias
hidden_sig=np.append(hidden_sig,1)
output=np.dot(hidden_sig,weight_output)
#predicted output
predicted=sigmoid(output)
print(predicted)

### plot loss function ###
plt.plot(np.arange(len(loss)),loss)
plt.show()
