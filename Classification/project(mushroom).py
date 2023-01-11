#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv('mushrooms.csv')


# In[3]:


data.head()


# In[4]:


data['stalk-surface-below-ring'][3]


# In[ ]:





# In[5]:


[data.columns]


# In[6]:


data.info()


# In[7]:


x=data['class'].map({'p':0,'e':1})
x


# In[8]:


y=data.loc[:,data.columns!='class']
y=pd.get_dummies(y)
y.head()


# In[9]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[10]:


final={}
train_x, test_x, train_y, test_y= train_test_split(y,x,test_size=0.4, random_state=10)
train_x.shape, test_x.shape, train_y.shape, test_y.shape


# In[ ]:





# ### LogisticRegression

# In[11]:


model1= LogisticRegression()
param1={"C":np.logspace(-3,3,7), "penalty":["l1","l2"],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
model1_GCV= GridSearchCV(model1,param1, cv=10, verbose=0)
model1_GCV.fit(train_x, train_y)


# In[12]:


model1_GCV.best_estimator_


# In[13]:


p1=model1_GCV.predict(test_x)


# In[14]:


final[type(model1).__name__]=accuracy_score(p1, test_y)


# In[15]:


accuracy_score(p1, test_y)


# ### RandomForestClassifier

# In[16]:


model2= RandomForestClassifier()
param2={'criterion':['gini', 'entropy'],
        'n_estimators':[100,140,180,200,300],
        'max_depth':[1,2,3],
        'bootstrap':[True, False]
       }
model2_GCV= GridSearchCV(model2,param2, cv=10, verbose=0)
model2_GCV.fit(train_x, train_y)


# In[17]:


model2_GCV.best_estimator_


# In[18]:


p2=model2_GCV.predict(test_x)


# In[19]:


final[type(model2).__name__]=accuracy_score(p2, test_y)


# In[20]:


accuracy_score(p2, test_y)


# ### AdaBoostClassifier

# In[21]:


model3= AdaBoostClassifier()
param3= {'base_estimator':[DecisionTreeClassifier(max_depth=1)],
         'n_estimators':[100,140,180,200,300],
         'learning_rate':[0.01, 0.1, 1, 10,100 ],
         'algorithm':['SAMME', 'SAMME.R']
        }
model3_GCV= GridSearchCV(model3,param3, cv=10, verbose=0)
model3_GCV.fit(train_x, train_y)


# In[22]:


model3_GCV.best_estimator_


# In[23]:


p3=model3_GCV.predict(test_x)
final[type(model3).__name__]=accuracy_score(p3, test_y)
accuracy_score(p3, test_y)


# ### GradientBoostingClassifier

# In[24]:


model4 =GradientBoostingClassifier()
param4= {'max_depth':[1,2,3],
         'n_estimators':[100,140,180,200,300],
         'learning_rate':[0.01, 0.1, 1, 10,100 ]
        }
model4_GCV= GridSearchCV(model4,param4, cv=10, verbose=0)
model4_GCV.fit(train_x, train_y)


# In[25]:


model4_GCV.best_estimator_


# In[26]:


p4=model4_GCV.predict(test_x)
final[type(model4).__name__]=accuracy_score(p4, test_y)
accuracy_score(p4, test_y)


#  ### SVC

# In[27]:


model5= SVC()
param5= {'C':[0.05,0.1,1,10,100],
         'gamma':[0.05,0.1,1,10,100]
        }
model5_GCV= GridSearchCV(model5,param5, cv=10, verbose=0)
model5_GCV.fit(train_x, train_y)


# In[28]:


model5_GCV.best_estimator_


# In[29]:


p5=model5_GCV.predict(test_x)
final[type(model5).__name__]=accuracy_score(p5, test_y)
accuracy_score(p5, test_y)


# ### KNeighborsClassifier

# In[30]:


model6=KNeighborsClassifier()
model6.fit(train_x, train_y)


# In[31]:


p6=model6.predict(test_x)
final[type(model6).__name__]=accuracy_score(p6, test_y)
accuracy_score(p6, test_y)


# In[32]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


final


# In[34]:


plt.figure(figsize=[8,6])
plt.bar(final.keys(), final.values())
plt.xticks(rotation=80,fontsize=15);


# In[35]:


final


# In[ ]:




