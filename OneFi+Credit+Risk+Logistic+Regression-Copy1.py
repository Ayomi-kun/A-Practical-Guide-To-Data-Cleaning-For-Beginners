
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


#Import Data Sets

test_demo = pd.read_csv('testdemographics.csv')
test_perf = pd.read_csv('testperf.csv')
test_prevloans = pd.read_csv('testprevloans.csv')
train_demo = pd.read_csv('traindemographics.csv')
train_perf = pd.read_csv('trainperf.csv')
train_prevloans = pd.read_csv('trainprevloans.csv')


# In[10]:


#Load and sort the dataset by customerid, and then check length of dataset

train_demo = train_demo.sort_values(by='customerid', ascending = 1)
len(train_demo)


# In[11]:


#Get information on DataSet
train_demo.info()


# In[12]:


#check for duplicates in dataset
train_demo[train_demo.duplicated(['customerid'],keep = False)]


# In[13]:


#Drop one of the duplicates and print length of dataset 

train_demo_nodups =train_demo.drop_duplicates(['customerid'], keep='last')
len(train_demo_nodups)


# In[14]:


train_perf = train_perf.sort_values(by='customerid', ascending = 1)
len(train_perf)


# In[15]:


train_perf.info()


# In[16]:


train_perf[train_perf.duplicated(['customerid'],keep=False)]
# The results reveal the duplicates


# In[17]:


print("Size of NoDups Train Demographics dataset", len(train_demo_nodups))
print("Size of NoDups Train Loan Perfomance dataset", len(train_perf))


# In[18]:


train_demo_perf = pd.merge(train_demo_nodups, train_perf, on='customerid', how='right')


# In[19]:


len(train_demo_perf)


# In[20]:


train_demo_perf.info()


# In[21]:


#sort dataset by customer ID and print out length
train_prevloans = train_prevloans.sort_values(by='customerid', ascending = 1)
len(train_prevloans)


# In[22]:


train_prevloans.info()


# In[23]:


#check for dupliates with SystemloanID and Customer ID
train_prevloans[train_prevloans.duplicated(['systemloanid'],keep=False)]


# In[24]:


train_demo_prev = pd.merge(train_demo_nodups, train_prevloans, on='customerid', how='right')


# In[25]:


train_demo_prev.info()


# In[26]:


train_demo_perf.info()


# In[27]:


train_demo_perf.bank_name_clients.unique()


# In[28]:


train_demo_perf.bank_account_type.unique()


# In[29]:


train_demo_perf.bank_branch_clients.unique()


# In[30]:


train_demo_perf.employment_status_clients.unique()


# In[31]:


train_demo_perf.level_of_education_clients.unique()


# In[32]:


train_demo_perf[train_demo_perf.duplicated(['customerid'],keep=False)]


# In[33]:


train_demo_perf.info()


# In[34]:


train_demo_perf = train_demo_perf.drop(['customerid','birthdate','systemloanid','approveddate','creationdate','referredby','bank_branch_clients'], axis =1)


# In[35]:


train_demo_perf.info()


# In[36]:


train_demo_perf.isnull().sum()


# In[37]:


train_demo_perf = train_demo_perf.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[38]:


train_demo_perf.isnull().sum()


# In[39]:


train_demo_perf['employment_status_clients'].unique()


# In[40]:


train_demo_perf.head(100)


# In[41]:


train_demo_perf.isnull().sum()


# In[42]:


train_demo_perf['employment_status_clients'].unique()


# In[43]:


train_demo_perf['bank_account_type'].unique()


# In[44]:


train_demo_perf['level_of_education_clients'].unique()


# In[45]:


train_demo_perf['bank_name_clients'].unique()


# In[46]:


values = {"good_bad_flag": {"Bad":0,"Good":1}, "employment_status_clients":{"Student":1,"Unemployed":2,"Self-Employed":3,"Retired":4,"Permanent":5,"Contract":6}, "bank_account_type":{"Savings":1,"Current":2,"Other":3}, 
          "level_of_education_clients":{"Primary":1,"Secondary":2,"Graduate":3,"Post-Graduate":4},"bank_name_clients":{'GT Bank':1, 'Standard Chartered':2, 'First Bank':3, 'Access Bank':4,
       'Diamond Bank':5, 'EcoBank':6, 'Stanbic IBTC':7, 'Sterling Bank':8,
       'Skye Bank':9, 'UBA':10, 'Zenith Bank':11, 'FCMB':12, 'Fidelity Bank':13,
       'Wema Bank':14, 'Union Bank':15, 'Keystone Bank':16, 'Heritage Bank':17,
       'Unity Bank':18}}
train_demo_perf.replace(values, inplace=True)


# In[47]:


train_demo_perf.head()


# In[48]:


corr_matrix = train_demo_perf.corr()


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[62]:


f, ax = plt.subplots(figsize=(10,4))
sns.heatmap(corr_matrix, linewidths=2.0, ax=ax, annot=True)
ax.set_title('Correlation Matrix')


# In[63]:


train_demo_perf["amount_total"] = train_demo_perf["loanamount"]* train_demo_perf["totaldue"] 
train_demo_perf["acc_number"] = train_demo_perf["loannumber"]* train_demo_perf["bank_account_type"] 
train_demo_perf["status_acc"] = train_demo_perf["bank_account_type"]* train_demo_perf["employment_status_clients"] 
train_demo_perf["number_amount"] = train_demo_perf["loanamount"]* train_demo_perf["loannumber"] 
train_demo_perf["amount_term"] = train_demo_perf["loanamount"]* train_demo_perf["termdays"] 


# In[72]:


real_traindata = train_demo_perf[["amount_total","acc_number","status_acc","number_amount","amount_term"]]


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[64]:


train_data = train_demo_perf.drop('good_bad_flag',axis=1)


# In[73]:


x_train, x_test, y_train, y_test = train_test_split(real_traindata, train_demo_perf['good_bad_flag'],test_size=0.33, random_state=1)


# In[74]:


rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
rf.fit(x_train, y_train)


# In[75]:


y_predict = rf.predict(x_test)
accuracy_score(y_test, y_predict)


# In[68]:


from sklearn.linear_model import LogisticRegression


# In[57]:


LR = LogisticRegression(random_state = 1)


# In[76]:


LR.fit(x_train,y_train)


# In[77]:


y_pred = LR.predict(x_test)
accuracy_score(y_test, y_pred)

