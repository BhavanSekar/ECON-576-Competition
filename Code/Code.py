#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
    #if count of unique values is less than 200 consider it to be categorical
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


# In[4]:


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


# In[5]:


#For nominal variables, create dummy features with 0 and 1 as values to avoid error in linear regression results
#This line of code creates a dummy variable for all nominal varables that is stored in a list.
df_d=pd.get_dummies(data=df, columns=columns_nominal_variables, dtype=np.int64)
#As new columns for dummy variables are created at the end of the dataframe, add the output column to the end of the df.
df_d["Output_Sale_Price"]=df_d["SalePrice"]
df_d=df_d.drop("SalePrice", axis='columns')
#Preview df_d. This is the final dataset before we proceed with modelling steps
df_d.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#set seed for replicable results. This way the same random numbers would be generated for test and train split.
import random
random.seed(11)
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

#Split into training and test sets. test set would be 10% the size of the dataset. Off the top of my head.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
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


# In[7]:


#Determine r2
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
#Other way to do the same thing
print(regr.score(X_test, y_test))


# In[8]:


#We get a reasonable r2 of 0.8704 on the test set!
#Now, train a new model on the entire training data available and predict the actual unseen test set(Validation) values.
#Process the test set the same way it was done earlier for the original train dataset
#This model is decent enough so we ignore paramter tuning for now. The model can be improved futher, later.


# In[9]:


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
#    1          0            0              1          0     
#    0          1            0              0          1
#    0          0            1              
#We don't want that. So remove features that are not seen in both of the columns.
for i in X.columns:
    if i not in df_d_val.columns:
        X=X.drop(i, axis="columns")
for i in df_d_val.columns:
    if i not in X.columns:
        df_d_val=df_d_val.drop(i, axis="columns")


# In[10]:


#train a new model on the entire training data available and predict the actual unseen test set(Validation) values.
regr_final = LinearRegression()
regr_final.fit(X, y)
#predict the scores for the actual validation/test set
y_val_pred=regr_final.predict(df_d_val)
#Eureka! This cannot be verified by us. All hail Frank, the TA.
print(y_val_pred)


# In[11]:


#Next task is to improve the model accuracy by using complex models(polynomial/lasso or ridge regression, CNN)
#Might have to make use of cross validation methods(GridsearchCV) and perform model tuning.


# In[53]:


#DO NOT RUN THIS AND THE NEXT CELL IF TIME IS SPARSE. TAKES A LONG TIME TO RUN.
#MODEL ACCURACY IS FURTHER IMPROVED IN LASSO REGRESSION. NEURAL NETWORK IS NOT REALLY USEFUL HERE.
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
#Scale the data as regularization methods in CNN are sensitive to scale/ranges of features.
scaler = MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#hyperparameters tuned until convergence by random variation in test train split. Change only if accuracy improves.
params = { 'hidden_layer_sizes' : [10,10],
    'activation' : 'relu', 'solver' : 'adam',
    'alpha' : 0.13, 'batch_size' : 10,
    'random_state' : 0, 'tol' : 0.0001,
    'nesterovs_momentum' : False,
    'learning_rate' : 'constant',
    'learning_rate_init' : 0.008,
    'max_iter' : 2500, 'shuffle' : True,
    'n_iter_no_change' : 50, 'verbose' : False }
net = MLPRegressor(**params)
net.fit(X_train_scaled, y_train)
y_pred_nn=net.predict(X_test_scaled)
print(r2_score(y_test, y_pred_nn))
#Accuracy improves by 0.5. But meh. Not worth it.


# In[13]:


#fit the neural net model on the whole dataset and predict y on actual test/validation set
scaler_full_data = MinMaxScaler()
X_scaled=scaler_full_data.fit_transform(X)
df_d_val_scaled=scaler_full_data.transform(df_d_val)
#fit the same model
params = { 'hidden_layer_sizes' : [10,10],
    'activation' : 'relu', 'solver' : 'adam',
    'alpha' : 0.13, 'batch_size' : 10,
    'random_state' : 0, 'tol' : 0.0001,
    'nesterovs_momentum' : False,
    'learning_rate' : 'constant',
    'learning_rate_init' : 0.008,
    'max_iter' : 2500, 'shuffle' : True,
    'n_iter_no_change' : 50, 'verbose' : False }
net_final = MLPRegressor(**params)
net_final.fit(X_scaled, y)
y_val_pred_nn=net_final.predict(df_d_val_scaled)


# In[52]:


print(y_val_pred_nn)


# In[62]:


#Next alternative is to try regression with regularization methods (Rigdge/Lasso)
#Fit lasso as lasso is better than ridge if the data has a lot of features. Ridge sets the penalty term to 0.
from sklearn.linear_model import Lasso
#Increaseing or decreasing the alpha by 1 would result in lesser r2. Model is converged at a test score of 91.24.
las = Lasso(alpha = 61)
las.fit(X_train_scaled, y_train)
#Predict y on local test set
y_pred_las=las.predict(X_test_scaled)
print(r2_score(y_test, y_pred_las))
#Lasso gives the best accuracy. A whopping 91.24! The accuracy on the actual test set should be 91.5 approximately
#Lasso just beat CNN. The mother of all algorithms! Hoping to probe more given time and inclination.
las_val = Lasso(alpha = 61)
las_val.fit(X_scaled, y)
y_val_pred_las=las_val.predict(df_d_val_scaled)
#print(y_val_pred_las)
df_submission=pd.DataFrame(data=y_val_pred_las, columns=["Predicted_Sale_Price_For_Test_Set"], index=None)
print(df_submission)
#Uncomment the next lines to have a look at the difference in predictions in different models.
#print(" ")
#print(y_val_pred_nn)
#print(" ")
#print(y_val_pred)

# determining the name of the file for submission
file_name = 'ECON_Competition_Submission.xlsx'
 
# saving the excel
df_submission.to_excel(file_name)
#Find the submission file in the same directory!


# In[ ]:




