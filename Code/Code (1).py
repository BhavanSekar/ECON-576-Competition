#!/usr/bin/env python
# coding: utf-8

# In[50]:


#DATA PREPROCESSING FOR LINEAR REGRESSION
import numpy as np
import pandas as pd
df=pd.read_csv("salepricedata.csv")
#Lookout for categorical variables
#List to store variables of different types
columns_categorical_variables=[]
columns_nominal_variables=[]
columns_ordinal_variables=[]
columns_continuous_variables=[]
#Loop across all columns and consider any column with less than 200 unique value as a categorical feature.
#Print report combined with list updation of categorical variables
print("Report of all categorical variables")
for i in df.columns:
    #if count of unique values is less than 200 consider it to be catoegorical
    if len(df[i].unique())<200:
        #if the first element of the unique value is a number or a float, consider it an ordinal data
        if type(df[i].unique()[0])==np.int64 or type(df[i].unique()[0])==np.float64:
            print("variable type=ordinal")
            columns_ordinal_variables.append(i)
            #For columns having ordinal data fill na values as 0.
            df[i]=df[i].fillna(0)
        #else consider it nominal data    
        else:
            print("variable type=nominal")
            columns_nominal_variables.append(i)
            #For columns having nominal data fill na values as "NULL".This will be considered as an additional category.
            df[i]=df[i].fillna("NULL")
        #Update categorical variables. Might be of use. Print categorical variables and see for any misclassifications.    
        columns_categorical_variables.append(i)
        print("column=", i)
        print("unique values=", df[i].unique())
        print("count of unique values=", len(df[i].unique()))    
        #For output readability. Ignore the next two prints.    
        print(" ")
        print(" ")
#Print the nominal and ordinal variables
print("nominal variables")
print(columns_nominal_variables)
print("ordinal variables")
print(columns_ordinal_variables)


# In[51]:


#Look at continuous variables. If a feature has more than 200 unique values it is continuous.
for i in df.columns:
    if len(df[i].unique())>200:
        #again for continuous variables, 0 is a good value to replace for null
        df[i]=df[i].fillna(0)
        print("column=", i)
        print("unique values=", df[i].unique())
        print("count of unique values=", len(df[i].unique()))
        print(" ")
        #Append all continous variables except the target
        if i!="SalePrice":
            columns_continuous_variables.append(i)


# In[52]:


#For nominal variables, create dummy features with 0 and 1 as values to avoid error in linear regression results
#This line of code creates a dummy variable for all nominal varables that is stored in a list.
df_d=pd.get_dummies(data=df, columns=columns_nominal_variables, dtype=np.int64)
#As new columns for dummy variables are created at the end of the dataframe, add the output column to the end of the df.
df_d["Output_Sale_Price"]=df_d["SalePrice"]
df_d=df_d.drop("SalePrice", axis='columns')
#Preview df_d. This is the final dataset before we proceed with modelling steps
df_d.head()


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#set seed for replicable results. This way the same random numbers would be generated for test and train split.
import random
random.seed(10)
#Drop the id column as it is not meant to be a feature for prediction.
df_d=df_d.drop("Id", axis='columns')
#The last column is the output variable. Rest all are features
X = df_d.iloc[:, :-1]
y = df_d.iloc[:, -1]
#Split into training and test sets

###### AS WE DO NOT HAVE THE OUTPUT VARIABLE IN TEST SET, WE ASSUME THE TRAINING SET TO BE THE SUPERSET.
###### THE DATAFRAME IMPORTED HERE IS THE TRAIN.CSV UPLOADED IN BRIGHTSPACE RENAMED AS SALEPRICEDATA.CSV.
###### THE TEST SET FROM BRIGHTSPACE WOULD BE THE VALIDATION OR THE FINAL UNSEEN TEST SET.
###### WE WILL PERFORM MODEL EVALUATION/ CROSS VALIDATION ON OUR LOCAL TEST TEST.
###### THE BEST POSSIBLE RESULT (r^2) WE SEE ON OUR LOCAL TEST WOULD ALSO HAVE TO BE REPLICATED ON THE UNSEEN
###### TEST SET. HOPEFULLY.

#Split into training and test sets. test set would be 30% the size of the dataset. Off the top of my head.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#Fit simple regression model.
regr = LinearRegression()
regr.fit(X_train, y_train)
#Predict y on local test set
y_pred=regr.predict(X_test)
#Preview predicted sale prices
print("Predicted Prices")
print(y_pred)
print(" ")
print("Actual Prices")
print(y_test)


# In[54]:


#Determine r2
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
#Other way to do the same thing
print(regr.score(X_test, y_test))


# In[55]:


#We get a reasonable r2 of 0.8541 on the test set!
#Now, train a new model on the entire training data available and predict the actual unseen test set(Validation) values.
#Process the test set the same way it was done earlier for the original train dataset
#This model is decent enough so we ignore paramter tuning for now. The model can be improved futher, later.


# In[56]:


#DATA PREPROCESSING FOR ACTUAL TEST/VALIDATION SET
import numpy as np
import pandas as pd
df_val=pd.read_csv("test_val.csv")
#The columns that are categorical do not change. Estimating it again might result in inconsistent results.
columns_categorical_variables_val=columns_categorical_variables
columns_nominal_variables_val=columns_nominal_variables
columns_ordinal_variables_val=columns_ordinal_variables
columns_continuous_variables_val=columns_continuous_variables
#For all categorical and continuous variables fillna based on if they are ordinal or nominal(0,NULL) pr continuous(0)
for i in columns_ordinal_variables_val:
        df_val[i]=df_val[i].fillna(0)
for i in columns_nominal_variables_val:
        df_val[i]=df_val[i].fillna("NULL")
for i in columns_continuous_variables_val:
        df_val[i]=df_val[i].fillna(0) 
#create dummy variables for nominal variables
df_d_val=pd.get_dummies(data=df_val, columns=columns_nominal_variables_val, dtype=np.int64)
#drop ID as it is not meant to exist. No.
df_d_val=df_d_val.drop("Id", axis="columns")
df_d_val
#Equalize the feature vectors for test and train set: If a category does not exist in any column a dummy column might be
#missing and the model collapses.
#train_col  test_col
# 1           1
# 2           2
# 3
#would result in
#train_col_1 train_col_2 train_col_3    test_col_1 test_col_1
     1          0            0              1          0     
     0          1            0              0          1
#    0          0            1              
#We don't want that. So remove features that are not seen in both of the columns.
for i in X.columns:
    if i not in df_d_val.columns:
        X=X.drop(i, axis="columns")
for i in df_d_val.columns:
    if i not in X.columns:
        df_d_val=df_d_val.drop(i, axis="columns")


# In[57]:


#train a new model on the entire training data available and predict the actual unseen test set(Validation) values.
regr_final = LinearRegression()
regr_final.fit(X, y)
#predict the scores for the actual validation/test set
y_val_pred=regr_final.predict(df_d_val)
#Eureka! This is the submission. This is result. This is everything. This cannot be verified by us. All hail Frank, the TA.
print(y_val_pred)


# In[ ]:


#Next task is to improve the model accuracy by using complex models(polynomial/lasso or ridge regression, CNN)
#Might have to make use of cross validation methods(GridsearchCV) and perform model tuning.


# In[ ]:





# In[ ]:





# In[ ]:




