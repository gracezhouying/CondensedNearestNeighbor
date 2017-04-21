Goal:

Prototype selection is used to choose a small number of important points which best summarizes the training set. 
Condensed nearest neighbor data reduction algorithm is used to achieve this goal.

Dataset: 

MNIST dataset. Use 1NN for prediction.

CNN algorithm: 

Create a new prototype set and add a random point (from training set) to it. 
Pick any point from the original training set, use the prototype to fit the 1NN model and predict the point’s label. 
If the predicted label is right, then it can be left out of the prototype. 
If the predicted label is wrong, then add it to the prototype. 
Proceed through the training set, until the prototype size is M or no more points to add to prototype.

Error Rate Calculation: 

Use the prototype as training data to fit the KNeighborsClassifier (n=1) model. 
Calculate the error rate on test images and test labels. 
Repeat the prototype selection and fit model process for N times to get the error bars and confidence intervals. 

Result:

From the mean error rate, prototype selection improves over random selection.
