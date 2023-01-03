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
img = px.imshow(df.corr(),title="Correlation Plot of the Heat Failure Prediction")
#img.show()