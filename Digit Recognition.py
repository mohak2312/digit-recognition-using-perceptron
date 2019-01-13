import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
mnist_train='D:\PSU\Fall 18\ML\Assignment\mnist_train.csv'      # path for training and test data
mnist_test='D:\PSU\Fall 18\ML\Assignment\mnist_test.csv'        # path for training and test data

## Gloabal variables
global x,x_t,w,t,y,Sum,T_predicted,acc_training,target,target_t,acc_testing,predicted_digit

def get_weights():                                              # Generate a random values between -0.05 and 0.05
    w=[]                                                        # for 7850 weights    
    for i in range(10):
        a=[]
        for j in range(785):
            a.append(random.uniform( -0.05, 0.05 ))
        w.append(a)                                             # append the random generated value to list
    w=np.array(w)                                               # convert that list into numpy array 
    return w

def get_dataset(file):
    with file as f:
        csv_reader=csv.reader(f)                        # read the csv file of training and test data
        x=[]                                            # for extracting the input values and target value
        target=[]
        for row in csv_reader:
            i=0
            X=[1]   
            for col in row:                             # if its first element then it is target value
                if(i==0):                               
                    target.append(int(col))             # append target value to list
                    i=1
                else:                                   # else it is input values
                    X.append(float(col)/255)            # append input values to list 
            x.append(X)
    x=np.array(x)                                       # convert the both list into numpy array                                    
    print(x.shape)
    target=np.array(target)
    return x, target

def training_testing():                                 # training for 60000 data and testing for 10000 data
    
    e=[0.001,0.01,0.1]                                  # eata value for weight update(learning rate)   
    acc_training=[]                                     # list to store the accuracy after each epoch 
    acc_testing=[]
    b=1
    k=0
    for eta in e:                                       
        test_acc=[]                                             
        train_acc=[]
        if(k==0):   
            M=acc_calculate(x,target)                   # calculate the accuracy before training   
            acc=(M/60000)*100
            k=1
        print(acc)
        test_acc.append(acc)                            # append that accuracy to both traiing and test list to compare 
        train_acc.append(acc)
        print("caculateion for eta: ",eta)              
        for b in range(50):                             # set data for 50 epoch
            print("epoch number :",b)
            a=0
            for inp in x:                                   #Input from dataset                  
                Sum=(np.dot(w,np.transpose(inp)))           # checking each perceptron for prediction
                y,t=predicted_output(Sum,a)                 # predict the output
                if(np.array_equal(t,y)==False):             # if target is not equal to predicted outpiut then
                    update_weights(t,y,inp,eta)             # update the weights              
                a=a+1
            predicted_op=acc_calculate(x,target)            # after completing the each epoch caculate the accuracy 
            acc_train=(predicted_op/60000)*100              # for training data 
            print("train accuracy :",acc_train)             
            train_acc.append(acc_train) 
            predicted_op_test=acc_calculate(x_t,target_t)   # caculate the accuracy of testing data
            acc_test=(predicted_op_test/10000)*100
            print("Test accuracy :",acc_test)
            test_acc.append(acc_test)
            b=b+1
        caclulate_confusion_mat(x_t,target_t)               # After 50 epoch caclulate the confusion matrix for particuler eata
        acc_testing.append(test_acc)                        # append that accuracy to both traiing and test list to compare 
        acc_training.append(train_acc)
            
    return acc_training,acc_testing

def predicted_output(Sum,a):                                # function prediction
    y=np.zeros(10)                                          # intialize 10 perceptron with zero values
    t=np.zeros(10)                                          
    t[target[a]]=1                                          # set target value  for perceptron 
    for j in range(10):
        if (Sum[j]>0):                                      # if dot product of weights and input is positive then
            y[j]=1                                          # output is set to 1 else output is 0
        else:
            y[j]=0 
    return y,t

def update_weights(t,y,inp,eta):
    for l in range(len(y)):                                 # Update the weights
        w[l]=w[l]+(eta*(t[l]-y[l])*inp)

def acc_calculate(x,target):                                # function for caulating the accuracy
    a=0
    T_predicted=0
    for inp in x:                                           # accuracy calculation    
        Sum_1=(np.dot(w,np.transpose(inp)))
        index = np.argmax(Sum_1)                            # find a preddicted digit
        if(index==target[a]):                               # if dot product of weights and input of the perceptron is max  
            T_predicted= T_predicted+1                      # and target is match with it then increment the accuracy count
        a=a+1
                         
    return T_predicted

def caclulate_confusion_mat(x,target):                      # function to display confusion matrix 
    predicted_digit=[]      
    for inp in x:
        Sum_1=(np.dot(w,np.transpose(inp)))
        index = np.argmax(Sum_1)                            # find a preddicted digit
        predicted_digit.append(index)
    print(confusion_matrix(target, predicted_digit))        # display the confusion matrix for actual digit and predicted digit
    return 0    
    
w=get_weights()                                             # get the weights
file_1=open(mnist_train,'r')                                # open training file for reading
x, target=get_dataset(file_1)                               # get the target values and inputs
file_1.close()
file_2=open(mnist_test,'r')                                 # open the test data file for reading
x_t, target_t=get_dataset(file_2)                           # get the target and input values
file_2.close()
acc_training,acc_testing= training_testing()                # start training the perceptrons and testing it
print(acc_training,acc_testing)                             # print accuracy 
eta=[0.001,0.01,0.1]
for i in range(3):                                          # display the graph of epoch vs accuracy for each eta
    plt.xlabel('epoch')                                     # set x lable of graph
    plt.ylabel('Accuracy(%)')                               # set y lable of graph
    plt.plot(acc_testing[i])                                # ploat the accuracy on graph
    plt.plot(acc_training[i])
    plt.legend(['Accuracy on the test data','Accuracy on the training data'], loc='lower right')   
    plt.show()                                              # show the graph


    



    

        

