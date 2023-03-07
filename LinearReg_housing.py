#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #import pandas library


# In[2]:


data = pd.read_csv("E:/HousingData.csv") # read dataset...make sure to change path


# In[3]:


data.head()


# In[4]:


data_ = data.loc[:, ['LSTAT','MEDV']] #removing all columns except these two
data_.head(5)


# In[5]:


import matplotlib.pyplot as plt          #ploting data samples
data.plot(x='LSTAT', y='MEDV', style='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()


# In[6]:


X=pd.DataFrame(data['LSTAT']) # independent column
y=pd.DataFrame(data['MEDV'])  # dependent column


# In[7]:


from sklearn.model_selection import train_test_split    #spliting dataset in 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1) 


# In[8]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


from sklearn.linear_model import LinearRegression   #linear regession library
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[12]:


print(regressor.intercept_)   #y_intercept


# In[13]:


print(regressor.coef_) #value of slope. It is negative because we have a negative slope


# In[17]:


predictions = regressor.predict(X_test) # we will predict y_test(dependent) values based on x_test values(independent)


# In[25]:


predictions[:7] #predicted values for the first 7 samples


# In[27]:


y_test[:7] #actual values for first 7 samples


# In[30]:


import numpy as np   #result evaluation
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




