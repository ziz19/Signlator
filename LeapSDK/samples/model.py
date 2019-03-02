#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    df = pd.read_pickle(filename)
    return df

df = load_data('a.pkl')
char_a = pd.DataFrame(df)
char_a = char_a.drop(char_a.columns[0], axis=1)

train_set1, test_set1 = train_test_split(char_a, test_size=0.2, random_state=42)

df = load_data('s.pkl')
char_s = pd.DataFrame(df)
char_s = char_s.drop(char_s.columns[0], axis=1)

train_set2, test_set2 = train_test_split(char_s, test_size=0.2, random_state=42)

df = load_data('t.pkl')
char_t = pd.DataFrame(df)
char_t = char_t.drop(char_t.columns[0], axis=1)

train_set3, test_set3 = train_test_split(char_t, test_size=0.2, random_state=42)



# In[33]:


l_a = ['a'] * 800
#label_a = pd.DataFrame(l)

l_s = ['s'] * 800
#label_s = pd.DataFrame(l)

l_t = ['t'] * 800
#label_t = pd.DataFrame(l)

new_label = l_a + l_s + l_t

#new_label = label_a.append(label_s, ignore_index=True)
#new_label = new_label.append(label_t, ignore_index=True)

# fingers.join(label, rsuffix="result")
new_data = train_set1.append(train_set2, ignore_index=True)
new_data = new_data.append(train_set3, ignore_index=True)

# new_label = new_label.T
new_label


# In[35]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='newton-cg',multi_class='multinomial').fit(new_data, new_label)


# In[38]:


correct_a = ['s'] * 200
clf.score(test_set2, correct_a)


# In[ ]:




