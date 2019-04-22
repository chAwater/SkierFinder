#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy  as np

from glob import glob


# In[2]:


days =  sorted(
    glob('./huabei/wanlong/*'),
    key = lambda x: int( os.path.basename(x) )
)


# In[3]:


for day in days:
    print (os.path.basename(day) )
    files = sorted(
        glob( os.path.join(day,'*.pkl') ),
        key = lambda x: int( os.path.basename(x)[:-4] )
    )
    
    sum_df = pd.DataFrame()
    for file in files:
        df = pd.read_pickle(file)
        sum_df = sum_df.append(df, ignore_index=True)

    sum_df.astype(
        {
            'x1' : np.int16,
            'x2' : np.int16,
            'y1' : np.int16,
            'y2' : np.int16,
        }
    ).to_pickle( os.path.join('bulk',os.path.basename(day)+'.pkl') )
#     break




# In[4]:
days = sorted( [os.path.basename(i) for i in glob('./huabei/wanlong/*')] )

for idx,day in enumerate(days):
    data_df = pd.read_pickle('./bulk/'+day+'.pkl')
    data_df['ref_day'] = np.int16(idx)
    data_df.to_csv('./bulk/'+day+'.csv.gz',index=False)





