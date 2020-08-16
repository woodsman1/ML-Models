import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
import pandas as pd

'''link of the original code (https://github.com/Santara/ML-MOOC-NPTEL/blob/master/lecture1/ML-Anirban_Tutorial1.ipynb)'''


''''======================== Linear Regression Tutorial ============================================================='''

# creating data
# no_sample = 100
# x = np.linspace(-np.pi, np.pi, no_sample)
# y = x*0.5 + np.sin(x) + np.random.random(x.shape)
# # plt.scatter(x,y)
#
#
# # splitting data into training(70%), validataion(15%) and test sets(15%)
# rand_index =  np.random.permutation(no_sample)
#
# x_train = x[rand_index[0:70]]
# y_train = y[rand_index[0:70]]
#
# x_val = x[rand_index[70:85]]
# y_val = y[rand_index[70:85]]
#
# x_test = x[rand_index[85:]]
# y_test = y[rand_index[85:]]
#
#
# # training the linear reg model
#
# model = linear_model.LinearRegression()
#
# x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
# y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))
#
# model.fit(x_train_for_line_fitting, y_train_for_line_fitting)
#
# plt.scatter(x_train,y_train, color='black')
#
# # plt.plot(x.reshape(len(x),1), model.predict(x.reshape(len(x),1)), color='blue')
# # plt.show()
#
#
# # evaluating the model
# mean_val_error = np.mean((y_val - model.predict(x_val.reshape(len(x_val),1)))**2)
# mean_test_error = np.mean((y_test - model.predict(x_test.reshape(len(x_test),1)))**2)
#
# print("mean_val_error: ",mean_val_error,"\nmean_test_error: ", mean_test_error)

#------------------------- end -----------------------------------------------


'''======================== **Logistic Regression Tutorial ============================================================='''

# iris = datasets.load_iris()
#
# X = iris['data'][:100, :2]
# Y = iris['target'][:100]
#
# # splitting data into training(70%), validation(15%) and test sets(15%)
# no_sample = len(Y)
#
# rand_index = np.random.permutation(no_sample)
#
# no_train_sample = int(no_sample *0.7)
# x_train = X[rand_index[:no_train_sample]]
# y_train = Y[rand_index[:no_train_sample]]
#
# no_val_sample = int(no_sample *0.15)
# x_val = X[rand_index[no_train_sample: no_train_sample+no_val_sample]]
# y_val = Y[rand_index[no_train_sample: no_train_sample+no_val_sample]]
#
# x_test =X[rand_index[no_train_sample+no_val_sample:]]
# y_test =Y[rand_index[no_train_sample+no_val_sample:]]
#
#
# # splitting the data into two class for visualization
# x_class0 = np.asarray([x_train[i] for i in range(len(x_train)) if y_train[i]==0])
# y_class0 = np.zeros((x_class0.shape[0]),dtype=np.int)
# x_class1 = np.asarray([x_train[i] for i in range(len(x_train)) if y_train[i]==1])
# y_class1 = np.ones((x_class1.shape[0]),dtype=np.int)
#
# plt.scatter(x_class0[:,0], x_class0[:,1] , color = "red", label ='Class0')
# plt.scatter(x_class1[:,0], x_class1[:,1] , color = "blue", label ='Class1')
#
# plt.xlabel('sepal length')
# plt.ylabel("sepal width")
# plt.legend()
# plt.show()
#
#
# # creating the model of logistic reg
# model = linear_model.LogisticRegression(C=1e5)
#
# full_x = np.concatenate((x_class0, x_class1), axis=0 )
# full_y = np.concatenate((y_class0, y_class1), axis=0 )
# model.fit(full_x,full_y)
#
#
# # creating the mesh
# h=0.02
# x_min, x_max = full_x[:, 0].min()-0.5,  full_x[:, 0].max()+0.5
# y_min, y_max = full_x[:, 1].min()-0.5,  full_x[:, 1].max()+0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#
# Z = Z.reshape(xx.shape)
#
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# plt.scatter(x_class0[:, 0], x_class0[:, 1], c='red', edgecolors='k', cmap=plt.cm.Paired)
# plt.scatter(x_class1[:, 0], x_class1[:, 1], c='blue', edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.show()

#======================== **End of Logistic Reggression =============================================================


'''======================= **Decision tree Tutorial ===================================================================='''




