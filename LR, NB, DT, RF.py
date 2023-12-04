#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#sources code:
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
#https://stackoverflow.com/questions/70508252/how-to-use-gridsearchcv-with-a-pipeline-and-multiple-classifiers


# In[1]:


import pandas as pd
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report


# In[2]:


data_dirs = list(set(glob('/home/marilu/training_dfs/*.csv')) - 
                 set(glob('/home/marilu/training_dfs/*_ft.csv')) - 
                 set(glob('/home/marilu/training_dfs/*balanced.csv')))
data_dirs


# In[3]:


#in data_dirs[] you have to place the number of the place where political leaning is in the row, now it is at the last place in the row, which is 7. 
data = pd.read_csv(data_dirs[7])
data


# In[4]:


vec = TfidfVectorizer()
X_post = vec.fit_transform(data['post'])


# In[5]:


y_political = data['political_leaning']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_post, y_political, test_size=0.3, random_state=42)


# In[7]:


classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}


# In[8]:


for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Classification Report ({name}):")
    print(classification_report(y_test, y_pred))


# In[ ]:




