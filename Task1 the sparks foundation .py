#!/usr/bin/env python
# coding: utf-8

# ## **The Sparks Foundation Internship**
# 
# ## **Task-1: Predict the percentage of marks of an student based on the number of study                              hours**
# 
# ## **Name:- Rajesh Nandi**
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# Let's start the task and predict the score if a student study for 9.25 hrs/day.

# In[91]:


# Importing all required libraries 

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


# Reading data from the link

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
data.head(10)


# ## **Exploring the data**

# In[93]:


data.shape


# In[94]:


data.describe()


# In[95]:


data.isnull().sum()


# In[96]:


data.info()


# ## **Data Visualisation**

# In[97]:


data.plot(x='Hours', y='Scores', style='o', label="score",figsize=(9,5))  
plt.title('Hours vs scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Score of the students')
plt.legend()
plt.grid()
plt.show()


# In[98]:


data.hist()


# In[99]:


sns.pairplot(data)


# In[100]:


data.plot(x="Hours",y="Scores",kind="bar",figsize=(9,6),facecolor="red")
plt.title('Hours vs scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Score of the students')
plt.show()


# ### **Divide the data into Dependent and Independent variable and Graphical Representation**

# In[101]:


x= data.iloc[:,:-1].values  
y= data.iloc[:,1].values  


# In[102]:


plt.figure(figsize=(6,6))
plt.scatter(x,y, label="scores",color="red")
plt.xlabel("hours studied")
plt.ylabel("scores of the studxent")
plt.title("hours vs scores")
plt.legend()
plt.show


# ## **Split the Data into training and testing and fit the Linear Regression Algorithm**

# In[103]:


from sklearn.model_selection import train_test_split  
training_x, testing_x, training_y, testing_y = train_test_split(x, y,test_size=0.2, random_state=0) 


# In[104]:


from sklearn.linear_model import LinearRegression  
Lin = LinearRegression()  
Lin.fit(training_x,training_y) 


# ## **Plotting the Regression Line**

# In[105]:


plt.figure(figsize=(8,6))
plt.scatter(training_x,training_y,label="actual value",color="green")
plt.plot(training_x,Lin.predict(training_x),label="prediction line",color="red")
plt.title("hours vs scores")
plt.xlabel("hours")
plt.ylabel("scores")
plt.legend()
plt.show()


# ## **Prediction**
# 

# In[106]:


training_x


# In[107]:


testing_x


# In[108]:


pred_y=Lin.predict(testing_x)


# In[109]:


testing_y[3]


# In[110]:


pred_y[3]


# In[111]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': testing_y, 'Predicted': pred_y})  
df 


# In[112]:


Lin.coef_ , Lin.intercept_


# ## **predict the score if a student study for 9.25 hrs/day.**

# In[113]:


a=Lin.predict([[9.25]])
print(f"predicted score is: {round(a[0],2)} %")


# ## **Evaluating the model**
# ## **Find Accuracy and Error**

# In[114]:


from sklearn.metrics import r2_score,mean_squared_error
p=r2_score(testing_y,pred_y)
mse=mean_squared_error(testing_y,pred_y)
print("Accuracy:",p)
print("Mean Squard Error:",mse)


# In[115]:


from sklearn.metrics import r2_score,mean_absolute_error
p=r2_score(testing_y,pred_y)
mse=mean_absolute_error(testing_y,pred_y)
print("Accuracy:",p)
print("Mean Absolute Error:",mse)

