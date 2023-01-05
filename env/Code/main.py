import os
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)

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
plt.show()