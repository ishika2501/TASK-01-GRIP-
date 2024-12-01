#!/usr/bin/env python
# coding: utf-8

# ### THE SPARKS FOUNDATION INTERNSHIP
# 
# #### DATA SCIENCE AND BUSINESS ANALYTICS

# In[1]:


## **TASK-01**


# ### PREDICTION USING SUPERVISED MACHINE LEARNING 
# 
# ##### Predict the percentage of a student based on the number of study hours.
# 
# #### Data can be found at http://bit.ly/w-data
# 
# #### What will be the predicted score if a student studies for 9.25hr/day?
# 
# #### AUTHOR :- ISHIKA GOEL

# In[2]:


#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

print("hello world")


# In[3]:


#importing data 
dataset = pd.read_csv('dataset (2).csv') 


# In[4]:


#reading top datas
dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.tail()


# In[7]:


#plotting scores 
dataset.plot(x='Hours',y='Scores',style='+')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.show() 


# In[8]:


# EXTRACTING FEATURES AND LABELS OF DATASET
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# In[9]:


print(x)


# In[10]:


print(y)


# In[11]:


# SPLITTING DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 40)


# In[12]:


x_train


# In[13]:


# CREATING AND TRAINING THE MODEL
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train , y_train)


# In[14]:


# PREDICTION USING THE MODEL
y_predict = model.predict(x_test)


# In[15]:


print(y_predict)


# In[16]:


# Plotting the regression line
line = model.coef_*x+model.intercept_
line


# In[17]:


# Plotting for the test data
plt.scatter(x,y)
plt.plot(x, line);
plt.show()


# In[18]:


# IMPORTING LIBRARY TO CHECK ACCURACY
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test , y_predict)
print(score)


# In[19]:


# test with hours = 9.25
hr = np.array([9.25]).reshape(1, -1)
p= model.predict(hr)
print("Student studies for 9.25 hrs/day")
print(" Score of student = {}".format(p[0]))


# #### CONCLUSION

# #### If a student studies for 9.25 hours per day the he scores 91.8558.
# 
# ### THANK YOU!

# In[ ]:




