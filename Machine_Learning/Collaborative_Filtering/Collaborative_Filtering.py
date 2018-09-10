
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine


# In[2]:


# 导入数据
data=pd.read_csv(r'C:\Users\greeye\Desktop\user.txt',sep=',',encoding='utf-8')
data.index=['U1',"U2",'U3','U4']


# In[3]:


data


# In[11]:


# 1、 数据量进行中心化处理
new_data= data.apply(lambda x:x -x.mean(),axis= 1)


# In[12]:


new_data


# In[13]:


# 2 用0 填充数据
for i in new_data.index:
    new_data.loc[i,:] = np.nan_to_num(new_data.loc[i,:])


# In[14]:


new_data


# In[16]:


# 相似度计算
def sim(x,y,metric='pearsonr'):
    if metric == 'cos':
        return 1-cosine(x,y)
    else:
        return pearsonr(x,y)[0]


# In[17]:


# 计算 U1 用户与其他用户的相似度
sim_list= []
for i in new_data.index:
    sim_list.append(sim(new_data.loc[i,:].values,new_data.loc['U1',:]))


# In[18]:


sim_list


# In[23]:


data


# In[21]:


# 计算用户 U1 与 U3 的相似度
sim(new_data.loc['U1',:],new_data.loc['U2',:])


# In[22]:


# 计算用户 U1 与 U4 的相似度
sim(new_data.loc['U1',:],new_data.loc['U4',:])


# In[25]:


# 预测用户对没有购买过的物品的评分
pingfeng = (4*0.31+5*0.57)/(0.31+0.57)
pingfeng

