#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings


# In[2]:


warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()


# In[3]:


df = pd.read_csv("advertising.csv")


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
df


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe().T


# In[10]:


df.isnull().values.any()
df.isnull().sum()


# In[11]:


sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", kind="reg")


# In[12]:


# Histograms to check the normality assumption of the dependent variable (Sales)

df.hist(bins=20)


# In[13]:


sns.lmplot(x='TV', y='Sales', data=df)
sns.lmplot(x='Radio', y='Sales', data=df)
sns.lmplot(x='Newspaper',y= 'Sales', data=df)


# In[14]:


# Correlation Heatmap to check for multicollinearity among independent/dependent variables

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()


# In[15]:


# Model Preparation

X = df.drop('Sales', axis=1)
y = df[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)


# In[16]:


lin_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=df).fit()
print(lin_model.params, "\n")


# In[17]:


print(lin_model.summary())


# In[18]:


results = []
names = []
models = [('LinearRegression', LinearRegression())]
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)


# In[19]:


new_data = pd.DataFrame({'TV': [100], 'Radio': [50], 'Newspaper': [25]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[20]:


new_data = pd.DataFrame({'TV': [25], 'Radio': [63], 'Newspaper': [80]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[ ]:




