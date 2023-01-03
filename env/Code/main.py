import os
import numpy as np 
import pandas as pd 

df = pd.read_csv("Code/heart.csv")
print(df.head())
#print(df.dtypes)


string_col = df.select_dtypes(include="object").columns
df[string_col]=df[string_col].astype("string")
#print(df.dtypes)


