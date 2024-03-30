#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


#load the dataset
df=pd.read_csv(r"C:\Users\febin\Downloads\Loan_Data.csv")
df


# In[5]:


df.head()


# In[6]:


#checks the missing values
df.isnull().sum()


# In[7]:


# gives you information about the dataset
df.info()
# object dtype are the possible categorical features in our dataset.


# In[8]:


print(df['Gender'].value_counts())


# In[9]:


print(df['Married'].value_counts())


# In[10]:


#Data visualization using correlation matrix,it can be used only for the numerical data

#Data Imputation
df['Gender']=df['Gender'].fillna('Male')
df['Married']=df['Married'].fillna('Yes')


# In[11]:


df.isnull().sum()


# In[12]:


#finding the most frequent data in the dependents columns
print(df['Dependents'].value_counts())


# In[13]:


#Understand that most of them have 0 as their number of dependents.
print(df['Dependents'].value_counts())
df['Dependents']=df['Dependents'].fillna(0)


# In[14]:


#Finding the frequent value of the self employed column
print(df['Self_Employed'].value_counts())


# In[15]:


#It shows most of them are not employed
df['Self_Employed']=df['Self_Employed'].fillna('No')


# In[16]:


#Filling the missing values in loan_amount with mean of the loan amount
mean_loan=df['LoanAmount'].mean()
print(mean_loan)
df['LoanAmount']=df['LoanAmount'].fillna(mean_loan)


# In[17]:


#finding the frequent value in Loan_Amount_Term
print(df['Loan_Amount_Term'].value_counts())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(360.0) 


# In[18]:


#filling the missing values in the credit history
df['Credit_History']=df['Credit_History'].fillna(1)
df.isnull().sum()


# In[19]:


#Data preprocessing,since we use logistic regression requires evrything to be in algebraic.
#Loan id column can be dropped,as it does not carries any significant information in building a model.
df=df.drop(columns=['Loan_ID'],axis=1)
df.head()


# In[20]:


#converting the categorical data in the Gender column by mapping into 1:Male and 0:Female
for gender in df['Gender']:
    if gender=='Male':
        df['Gender']=df['Gender'].replace(to_replace='Male',value=1)
    else:
        df['Gender']=df['Gender'].replace(to_replace='Female',value=0)


# In[21]:


#converting the categorical data in the married column by mapping into 1:Yes and 0:No
for married in df['Married']:
    if married=='Yes':
        df['Married']=df['Married'].replace(to_replace='Yes',value=1)
    else:
        df['Married']=df['Married'].replace(to_replace='No',value=0)
#in dependents column
for dependents in df['Dependents']:
    if dependents=='3+':
        df['Dependents']=df['Dependents'].replace(to_replace='3+',value=3)


# In[22]:


#in education column
for education in df['Education']:
    if education=='Graduate':
        df['Education']=df['Education'].replace(to_replace='Graduate',value=1)
    else:
        df['Education']=df['Education'].replace(to_replace='Not Graduate',value=0)


# In[24]:


#in self-employed column
for self_employed in df['Self_Employed']:
    if self_employed=='Yes':
        df['Self_Employed']=df['Self_Employed'].replace(to_replace='Yes',value=1)
    else:
        df['Self_Employed']=df['Self_Employed'].replace(to_replace='No',value=0)
df.head()


# In[25]:


#in Property_Area
for property_area in df['Property_Area']:
    if property_area=='Urban':
        df['Property_Area']=df['Property_Area'].replace(to_replace='Urban',value=0)
    elif property_area=='Semiurban':
        df['Property_Area']=df['Property_Area'].replace(to_replace='Semiurban',value=1)
    else:
        df['Property_Area']=df['Property_Area'].replace(to_replace='Rural',value=2)
df.head()      


# In[26]:


#in Loan Status
for loan_status in df['Loan_Status']:
    if loan_status=='Y':
        df['Loan_Status']=df['Loan_Status'].replace(to_replace='Y',value=1)
    else:
        df['Loan_Status']=df['Loan_Status'].replace(to_replace='N',value=0)
correlation_matrix=df.corr().round(2)
fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(correlation_matrix,cmap="YlGnBu",square=True,annot=True,ax=ax,)


# In[27]:


x=pd.DataFrame(np.c_[df['Gender'],df['Married'],df['Dependents'],df['Education'],df['Self_Employed'],df['ApplicantIncome'],df['CoapplicantIncome'],df['Loan_Amount_Term'],df['Credit_History'],df['Property_Area']],columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','Loan_mount_Term','Credit_History','Property_Area'])
y=df['Loan_Status']
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()


# In[28]:


from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.25,random_state=9)
print(testy.shape)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
reg_model=DecisionTreeClassifier(criterion="entropy",max_depth=7,min_samples_split=10)
reg_model.fit(trainx,trainy)


# In[30]:


predictions=reg_model.predict(testx)
print(predictions[0:10])


# In[31]:


from sklearn.metrics import accuracy_score
score=accuracy_score(testy,predictions)
print(score)


# In[32]:


#Defining the model with Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
reg_model=RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=8,min_samples_split=10,max_features=10)
reg_model.fit(trainx,trainy)
predictions=reg_model.predict(testx)
print("The predictions are:",predictions[0:10])
score=accuracy_score(testy,predictions)
print("The accuracy score is :",score)


# In[33]:


from sklearn.linear_model import LogisticRegression
reg_model=LogisticRegression()
reg_model=reg_model.fit(trainx,trainy)
train_score=reg_model.score(trainx,trainy)
print("The score is:",train_score)


# In[34]:


from sklearn.naive_bayes import GaussianNB
reg_model=GaussianNB()
reg_model.fit(trainx,trainy)
#Checking the model Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(testy,predictions)
print("The accuracy of the model is :",score)


# In[36]:


from joblib import dump


# In[37]:


dump(reg_model, 'loan_prediction.joblib')


# In[39]:


from joblib import load
loaded_model = load('loan_prediction.joblib')

