#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[26]:


A4_regression = pd.read_csv('AutoMPG.csv')
A4_regression.info()
corr = A4_regression.corr()
print(corr)
sns.heatmap(data = corr)


# In[27]:


A4_regression.head()


# In[28]:


X = A4_regression[[ 'Cylinders', 'Displacement', 'Weight', 
 'Acceleration','ModelYear']]
Y = A4_regression['MPG']

train = A4_regression[:(int((len(A4_regression)*0.8)))]
test = A4_regression[(int((len(A4_regression)*0.8))):]


# In[29]:



Regression1 = linear_model.LinearRegression()
train_x = np.array(train[[ 'Cylinders', 'Displacement', 'Weight', 
 'Acceleration','ModelYear']])
train_y = np.array(train['MPG'])
Regression1.fit(train_x,train_y)
test_x = np.array(test[[ 'Cylinders', 'Displacement',  'Weight', 
 'Acceleration','ModelYear']])
test_y = np.array(test['MPG'])


# In[30]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[31]:


R = r2_score(test_y,Y_pred)
M = np.sqrt(mean_squared_error(test_y,Y_pred))
print ('R² :',R)
print('Mean absolute error: %.2f' % np.mean(np.absolute(Y_pred - test_y)))
print('Mean sum of squares (MSE): %.2f' % np.mean((Y_pred - test_y) ** 2))
print('M', M)


# In[32]:


#Number 2 Algorithm 
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split


# In[33]:


A4_regression2 = pd.read_csv('AutoMPG.csv')


# In[34]:


A4_regression2.head()


# In[35]:


label1 = A4_regression2 [[ 'Cylinders', 'Displacement', 'Weight', 
 'Acceleration','ModelYear']]
label2 = A4_regression2 ['MPG']


# In[36]:


label1_train,label1_test, label2_train, label2_test = train_test_split(label1 ,label2, test_size=0.20)

reg = DecisionTreeRegressor()
reg.fit(label1_train, label2_train)


# In[37]:


prediction1 = reg.predict(label1_test)

KY =pd.DataFrame({'Actual':label2_test, 'Predicted':prediction1})


# In[38]:


from sklearn import metrics


# In[39]:


print('Mean Absolute Error:', metrics.mean_absolute_error(label2_test, prediction1))
print('Mean Squared Error:', metrics.mean_squared_error(label2_test, prediction1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(label2_test, prediction1)))


# In[ ]:





# In[ ]:




