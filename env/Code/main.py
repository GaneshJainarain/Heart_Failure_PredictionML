import os
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
import matplotlib 
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
acc_log=[]
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

df = pd.read_csv("Code/heart.csv")
print(df.head())
print("\n")
print(df.dtypes)

print("\n")
string_col = df.select_dtypes(include="object").columns
df[string_col] = df[string_col].astype("string")
print(df.dtypes)
print("\n")


string_col = df.select_dtypes("string").columns.to_list()
print(string_col)
print("\n")

num_col = df.columns.to_list()
print(num_col)

for col in string_col:
    num_col.remove(col)
num_col.remove("HeartDisease")
print("\n")


print(num_col)
print("\n")

#print(df.head())
print(df.describe().T)
img = px.imshow(df.corr(),title="Correlation Plot of the Heart Failure Prediction")
#img.show()

# Shows the Distribution of Heat Diseases with respect to male and female
fig=px.histogram(df, 
                 x="HeartDisease",
                 color="Sex",
                 hover_data=df.columns,
                 title="Distribution of Heart Diseases",
                 barmode="group")
#fig.show()

fig1 =px.histogram(df,
                 x="ChestPainType",
                 color="Sex",
                 hover_data=df.columns,
                 title="Types of Chest Pain"
                )
#fig1.show()

fig2 = px.histogram(df,
                 x="Sex",
                 hover_data=df.columns,
                 title="Sex Ratio in the Data")
#fig2.show()

fig3 = px.histogram(df,
                 x="RestingECG",
                 hover_data=df.columns,
                 title="Distribution of Resting ECG")
#fig3.show()
'''
plt.figure(figsize=(12,10))
sns.pairplot(df,hue="HeartDisease")
plt.title("Looking for Insights in Data")
plt.legend("HeartDisease")
plt.tight_layout()
plt.plot()
plt.show()
'''

plt.figure(figsize=(15,10))
for i,col in enumerate(df.columns,1):
    plt.subplot(4,3,i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(df[col],kde=True)
    plt.tight_layout()
    plt.plot()
#plt.show()

fig4 = px.box(df,y="Age",x="HeartDisease",title=f"Distribution of Age")
#fig4.show()

fig5 = px.box(df,y="RestingBP", x="HeartDisease",title=f"Distribution of RestingBP",color="Sex")
#fig5.show()

fig6 = px.box(df,y="Cholesterol", x="HeartDisease",title=f"Distribution of Cholesterol")
#fig6.show()

fig7 = px.box(df,y="Oldpeak",x="HeartDisease",title=f"Distribution of Oldpeak")
#fig7.show()

fig8 = px.box(df,y="MaxHR",x="HeartDisease",title=f"Distribution of MaxHR")
#fig8.show()

# Checking for Type of data
print(df.info())
print("\n")
# Checking for NULLs in the data
print(df.isnull().sum())

# data
x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal
 
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])
 
scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])
 
scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])
 
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))
ax1.set_title('Before Scaling')
 
sns.kdeplot(x['x1'], ax = ax1, color ='r')
sns.kdeplot(x['x2'], ax = ax1, color ='b')
ax2.set_title('After Robust Scaling')
 
sns.kdeplot(robust_df['x1'], ax = ax2, color ='red')
sns.kdeplot(robust_df['x2'], ax = ax2, color ='blue')
ax3.set_title('After Standard Scaling')
 
sns.kdeplot(standard_df['x1'], ax = ax3, color ='black')
sns.kdeplot(standard_df['x2'], ax = ax3, color ='g')
ax4.set_title('After Min-Max Scaling')
 
sns.kdeplot(minmax_df['x1'], ax = ax4, color ='black')
sns.kdeplot(minmax_df['x2'], ax = ax4, color ='g')
#plt.show() 


df[string_col].head()
for col in string_col:
    print(f"The distribution of categorical values in the {col} is : ")
    print(df[col].value_counts())
    print("\n")

# which will be used with Tree Based Algorthms
df_tree = df.apply(LabelEncoder().fit_transform)
print(df_tree.head())

print("\n")
print("-----")
## Creating one hot encoded features for working with non tree based algorithms 
df_nontree = pd.get_dummies(df,columns = string_col,drop_first=False)
print(df_nontree.head())

# Getting the target column at the end
target = "HeartDisease"
y = df_nontree[target].values
df_nontree.drop("HeartDisease",axis = 1,inplace = True)
df_nontree =pd.concat([df_nontree,df[target]],axis = 1)
print(df_nontree.head())

