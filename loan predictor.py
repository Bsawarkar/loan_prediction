
# coding: utf-8

# In[42]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


#  reading files from local machine 

train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")


# In[44]:


# cheking data types

test.info()


# In[45]:


#checking top 5 rows
train.head()


# In[46]:


test.head()


# # adding new column named Loan_status to test set 

# as the test dataset is not having this column

# In[47]:


test['Loan_Status'] = 0


# In[48]:


test.head()


# # now concatenating the train and test dataset into one

# In[49]:


x = pd.concat([train,test],axis = 0)


# # now we will convert the loan id column into numeric form

# as loan id has 'LP' as prefix, we will remove this for the model understandable form

# In[50]:


x['Loan_ID']=x['Loan_ID'].str.extract('(\d+)').astype(np.float)


# In[51]:


x.head(2)


# # now we will check if there are any null values using seaborn heatplot

# yellow line shoes the places where it is empty 

# In[52]:


sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# checking null values and analyse majority of gender in the sector

# In[53]:


x['Gender'].isnull().value_counts() # null value count


# In[54]:


x['Gender'].value_counts()


# In[55]:


# here we can see that only 24 values are empty and majoirty of the person apllying are male therefore imputing male in the null places.


# In[56]:


x['Gender']= x.Gender.map({'Male':0,'Female':1 , np.NaN : 0}).astype(np.float)


# In[57]:


x['Gender'].value_counts() # here we can see clearly that all the values have been filled with numerice form


# In[58]:


x['Married'].value_counts()


# In[59]:


x['Married'].isnull().value_counts() # as here we see that married are more therefore we will impute married in the empty spaces


# In[60]:


x['Married']= x.Married.map({'Yes':0,'No':1 , np.NaN : 0}).astype(np.float)


# In[61]:


x['Married'].value_counts()


# In[62]:


# now converting education into numeric form also  education is totally filled


# In[63]:


x['Education'].value_counts()


# In[64]:


x['Education']= x.Education.map({'Graduate':0,'Not Graduate':1}).astype(np.float)


# In[65]:


x['Education'].value_counts()


# In[66]:


sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[67]:


x['Dependents'].isnull().value_counts()


# In[68]:


x['Dependents'].value_counts()


# In[69]:


# imputing 0 as the number of dependents


# In[70]:


x['Dependents']= x.Dependents.map({'0':0,'1':1 ,'2':2,'3+':3, np.NaN : 0}).astype(np.float)


# In[71]:


x['Dependents'].value_counts()


# In[72]:


x['Self_Employed'].isnull().value_counts() # nul value counts


# In[73]:


x['Self_Employed'].value_counts()


# In[78]:


# here we can see that all the nan values has less income also people with low income are eqully devided.


# In[40]:


#plt.(x['Education'], x['Property_Area'],alpha= 0.1)
#plt.show()


# In[80]:


# let us first convert  all the variable in numerice form to check correlation matrix


# In[81]:


sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[82]:


#x['Self_Employed']= x.Self_Employed.map({'No':0,'Yes':1}).astype(np.float) # here nan wii still remain nan values


# In[83]:


#x['Self_Employed'].isnull().value_counts()


# In[84]:


x['Self_Employed'].value_counts()


# In[85]:


x['Property_Area'].value_counts()


# In[86]:


x['Property_Area']= x.Property_Area.map({'Semiurban':0,'Urban':1,'Rural':2}).astype(np.float)


# In[87]:


x['Property_Area'].value_counts()


# In[91]:


x.corr()


# In[179]:


sns.countplot(x='Property_Area',hue='Credit_History',data=x)


# In[180]:


sns.countplot(x='Education',hue='Credit_History',data=x)


# In[181]:


sns.countplot(x='Education',hue='Dependents',data=x)


# In[182]:


sns.countplot(x='Credit_History',hue='Gender',data=x)


# In[92]:


# here we can see that the loan amount is havin high corelation with applicantincome


# In[93]:


x['Self_Employed']= x.Self_Employed.map({'No':0,'Yes':1 ,np.NaN : 0}).astype(np.float) # as self employed hase a ration of 8:1 for employed and not
# therefore directly imputing  self employed as yes for null values


# In[94]:


x['Self_Employed'].value_counts()


# In[95]:


# now self employes has been imputed


# In[97]:


plt.scatter(x['LoanAmount'], x['ApplicantIncome'],alpha=0.5)
plt.show()
# loan amount is on y axis


# In[98]:


# here we can see that loan amount is heavily concentrated in a particular range.


# In[99]:


x['LoanAmount'].isnull().value_counts()


# In[100]:


x['LoanAmount'].median()# imputing the median in null values


# In[101]:


x['LoanAmount'].mean()


# In[102]:


# here the data seems to be right skewed 


# In[103]:


def loan1(ln):
    l = ln
    if pd.isnull(l):
        return 126.0
    else: 
        return l


# In[104]:


x['LoanAmount'] = x['LoanAmount'].apply(loan1)


# In[105]:


# after this still median should not be changed


# In[106]:


x['LoanAmount'].median()


# In[107]:


x['LoanAmount'].mean()


# In[108]:


x['LoanAmount'].isnull().value_counts()


# In[109]:


# now we will handle the remaining data


# In[110]:


sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[111]:


x['Loan_Amount_Term'].value_counts()


# In[112]:


# here majority loan term are 360.0 therefore imputinh the same


# In[113]:


def term(ln):
    l = ln
    if pd.isnull(l):
        return 360.0
    else: 
        return l


# In[114]:


x['Loan_Amount_Term'] = x['Loan_Amount_Term'].apply(term)


# In[115]:


x['Loan_Amount_Term'].value_counts()


# In[116]:


# now loan amount term has also been imputed


# In[117]:


x['Credit_History'].isnull().value_counts()


# In[118]:


x['Credit_History'].value_counts() # majority has credit history as 1


# In[119]:


def ln(ln):
    l = ln
    if pd.isnull(l):
        return 1.0
    else: 
        return l


# In[120]:


x['Credit_History'] = x['Credit_History'].apply(ln)


# In[121]:


x['Credit_History'].value_counts()


# In[122]:


sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[124]:


x.head(5)


# In[125]:


# now spliting the data into the original train and test set


# In[126]:


x_trn = x.iloc[:614,:]
x_tst = x.iloc[614:,:]


# In[127]:


x_tst.info()


# In[128]:


x_trn['Loan_Status']= x_trn.Loan_Status.map({'N':0,'Y':1 }).astype(np.int)


# In[129]:


x_trn.head(10)


# In[130]:


x_trn_x = x_trn.iloc[:,1:-1] # splitting data into dependent and independent variable
#   x_trn_x independent variable


# In[131]:


x_trn_y = x_trn.iloc[:,12] # dependent variable


# In[132]:


x_trn_y.head(5)


# In[133]:


x_trn_x.head(3)


# In[134]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_trn_x, x_trn_y, test_size = 0.20 , random_state = 0)


# In[135]:


# scaling the data for bringing all variable in same range.
# please note it does not change the actual location in space of data points


# In[136]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[138]:


# applying random forest classifier


# In[139]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[140]:


y_pred = classifier.predict(X_test) # predicicting X_test to validate


# In[141]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[142]:


cm


# In[143]:


x_tst_final = x_tst.iloc[:,1:-1] # original test data


# In[144]:


x_tst_final.head(2)


# In[145]:


y_pred_final = classifier.predict(x_tst_final)


# In[147]:


#applying neual netword using keras


# In[148]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD


# In[149]:


model = Sequential()
model.add(Dense(4, input_shape=(11,), activation='sigmoid'))
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))


# In[150]:


model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])


# In[151]:


model.summary()


# In[152]:


x_trn.info()


# In[153]:


model.fit(x_trn_x,x_trn_y , epochs=40)


# In[154]:


x_tst_final.info()


# In[155]:


temp_class = model.predict(x_tst_final) > 0.5


# In[156]:


temp_class = pd.DataFrame(temp_class)


# In[157]:


def returnp(ln):
    l = ln
    if l == True :
        return 'Y'
    else: 
        return 'N'


# In[158]:


temp_class[0] = temp_class[0].apply(returnp)


# In[160]:


# applying logistin regression


# In[161]:


from sklearn.linear_model import LogisticRegression


# In[162]:


logmodel=LogisticRegression()


# In[163]:


logmodel.fit(X_train,y_train)


# In[164]:


predictions=logmodel.predict(X_test)


# In[165]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[166]:


# precission is 84 percent


# In[167]:


test.head()


# In[168]:


y_pred_final = pd.DataFrame(y_pred_final)


# In[169]:


####### for logistic regression #######################


# In[170]:


predict = logmodel.predict(x_tst_final)


# In[171]:


predict = pd.DataFrame(predict)


# In[172]:


# return processing


# In[173]:


def returnp(ln):
    l = ln
    if l == 1 :
        return 'Y'
    else: 
        return 'N'


# In[174]:


predict[0] = predict[0].apply(returnp)


# In[175]:


y_pred_final.head()


# In[176]:


# converting the file into dataframe
submission = pd.DataFrame({
        
        "Loan_ID": test["Loan_ID"],
        "Loan_Status":predict[0]
        } )
cols=["Loan_ID","Loan_Status"]
submission=submission.reindex(columns=cols)


# In[177]:


submission.to_csv('loan_int.csv',index = False) # saving the file in csv format

