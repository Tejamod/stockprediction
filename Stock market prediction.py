
# coding: utf-8

# In[1]:


import pandas as pd
import quandl
import math,datetime
import time
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing,svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import style
style.use('ggplot')
df=quandl.get("NSE/BAJAJ_AUTO")
print(df.head())


# In[2]:


print (df.tail())


# In[3]:


df.plot(kind='box',subplots=True,layout=(1,7),sharex=False,sharey=False)


# In[4]:


df.hist()


# In[5]:


scatter_matrix(df)
plt.show()


# In[6]:


print(df.tail())


# In[7]:


df ['OC_change']=(df['Close']-df['Open'])/df['Open']*100
df['HL_change']=(df['High']-df['Low'])/df['Low']*100
df=df[['Close','HL_change','OC_change']]
print(df.tail())


# In[8]:


forecast_col='Close'
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)


# In[9]:


print(df.tail())


# In[10]:


df.dropna(inplace=True)
print(df.tail())


# In[11]:


x=np.array(df.drop(['label'],1))
y=np.array(df['label'])
print(len(x),len(y))


# In[12]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)


# In[13]:


clf=LinearRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[14]:


x=x[:-forecast_out]
x_lately=x[-forecast_out:]
forecast_set=clf.predict(x_lately)
print(forecast_set)


# In[15]:


df['forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=time.mktime(last_date.timetuple())
one_day=86400
next_unix=last_unix+one_day
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
df['Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()

