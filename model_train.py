#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    df = pd.read_pickle(filename)
    return df

#i
i1 = load_data('./faetures/i1.pkl')
i2= load_data('./faetures/i2.pkl')
i3= load_data('./faetures/i3.pkl')
i4= load_data('./faetures/i4.pkl')


i = i1 + i2 + i3 + i4
i_df = pd.DataFrame(i)

train_set_i, test_set_i = train_test_split(i_df, test_size=0.2, random_state=42)
label_i = ['i'] * 16000
test_i =  ['i'] * 4000

#p
p1 = load_data('./faetures/p1.pkl')
p2 = load_data('./faetures/p2.pkl')
p3 = load_data('./faetures/p3.pkl')
p4 = load_data('./faetures/p4.pkl')

p = p1 + p2 + p3 + p4
p_df = pd.DataFrame(p)

train_set_p, test_set_p = train_test_split(p_df, test_size=0.2, random_state=42)
label_p = ['p'] * 16000
test_p =  ['p'] * 4000

#t
t1 = load_data('./faetures/t1.pkl')
t2= load_data('./faetures/t2.pkl')
t3= load_data('./faetures/t3.pkl')
t4= load_data('./faetures/t4.pkl')


t = t1 + t2 + t3 + t4
t_df = pd.DataFrame(t)

train_set_t, test_set_t = train_test_split(t_df, test_size=0.2, random_state=42)
label_t = ['t'] * 16000
test_t =  ['t'] * 4000

# ok
stop1 = load_data('./faetures/stop1.pkl')
stop2 = load_data('./faetures/stop2.pkl')
stop3 = load_data('./faetures/stop3.pkl')
stop4 = load_data('./faetures/stop4.pkl')

stop = stop1 + stop2 + stop3 + stop4
stop_df = pd.DataFrame(stop)

train_set_stop, test_set_stop = train_test_split(stop_df, test_size=0.2, random_state=42)
label_stop = ['stop'] * 16000
test_stop =  ['stop'] * 4000



# In[44]:


new_label = label_p + label_i + label_t + label_stop
train_set = train_set_p.append(train_set_i, ignore_index=True)
train_set = train_set.append(train_set_t, ignore_index=True)
train_set = train_set.append(train_set_stop, ignore_index=True)



# In[45]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='newton-cg',multi_class='multinomial').fit(train_set, new_label)


# In[47]:


correct_p = ['p'] * 4000
correct_i = ['i'] * 4000
correct_t = ['t'] * 4000
correct_stop = ['stop'] * 4000

correct = correct_p + correct_i + correct_t + correct_stop

test_set = test_set_p.append(test_set_i, ignore_index=True)
test_set = test_set.append(test_set_t, ignore_index=True)
test_set = test_set.append(test_set_stop, ignore_index=True)

clf.score(test_set, correct)


# In[48]:


import pickle
filename = ('model_final.sav')
pickle.dump(clf, open(filename,'wb'))



# In[ ]:





# In[ ]:




