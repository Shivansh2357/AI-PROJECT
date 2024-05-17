

# In[9]:


#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
file_path= r'C:\Users\SHIVANSH GUPTA\Desktop\ProjectAIML\Life Expectancy Data.csv'


# In[11]:


df = pd.read_csv(file_path)
df


# In[12]:


plt.figure(figsize=(6,4))
sns.countplot(x='Status', data=df, palette='Set2')
plt.title('Status of the country')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


# In[13]:


df.info()


# In[14]:


#Checking if there are duplicate rows
df.duplicated().sum()


# In[15]:


df.describe()


# In[16]:


df.shape


# In[17]:


df.isna().sum()


# In[18]:


df.columns


# In[19]:


col_with_null_values = ['Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Polio', 'Total expenditure',
       'Diphtheria ', 'GDP', 'Population',' thinness  1-19 years', ' thinness 5-9 years','Income composition of resources', 'Schooling']
for i in col_with_null_values:
    mean = df[i].mean()
    df[i].fillna(value=mean, inplace = True)


# In[20]:


df.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)
df.head()


# In[21]:


X = df[['Adult_Mortality', 'Infant_Deaths', 'Alcohol',
        'HepatitisB', 'Measles', 'BMI', 'Under_Five_Deaths', 'Polio',
        'Diphtheria', 'HIV/AIDS']]
y = df['Life_Expectancy']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


model = LinearRegression()


# In[24]:


model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)


# In[27]:


rse=r2_score(y_test,y_pred)
print("R-Squared error:",rse)


# In[28]:


user_input = {
    'Adult_Mortality': float(input("Enter Mortality in adults: ")),
    'Infant_Deaths': int(input("Enter infants death: ")),
    'Alcohol': float(input("Alcohol: ")),
    'HepatitisB': int(input("HepatitisB: ")),
    'Measles': int(input("Measles: ")),
    'BMI': float(input("BMI: ")),
    'Under_Five_Deaths': int(input("Under five deaths: ")),
    'Polio': int(input("Polio: ")),
    'Diphtheria': int(input("Diphtheria: ")),
    'HIV/AIDS': float(input("HIV/AIDS: "))
}

user_input_df = pd.DataFrame([user_input])
predicted_output = model.predict(user_input_df)
print("Predicted Life Expectancy:", predicted_output[0])

