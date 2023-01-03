## When dealing with machine learning problems, there are generally two types of data (and machine learning models):

- `Supervised data:` always has one or multiple targets associated with it.
- `Unsupervised data:` does not have any target variable.

A supervised problem is considerably easier to tackle than an unsupervised one. A problem in which we are required to predict a value is known as a supervised problem. 
For example, if the problem is to predict house prices given historical house prices, with features like presence of a hospital, school or supermarket, distance to nearest public transport, etc. is a unsupervised problem. 
Similarly, when we are provided with images of cats and dogs, and we know beforehand which ones are cats and which ones are dogs, and if the task is to create a model which predicts whether a provided image is of a cat or a dog, the problem is considered to be supervised.

## Here in this Dataset we have a Supervised Machine Learning Problem, For Heart Failure Prediction

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyper-lidipaemia or already established disease) need early detection and management wherein a machine learning model can be of great help


### Getting Started with our Data

The describe() function in pandas is very handy in getting various summary statistics.This function returns the count, mean, standard deviation, minimum and maximum values and the quantities of the data

```python

import numpy as np 
import pandas as pd 

df = pd.read_csv("Code/heart.csv")
print(df.head())

print(df.dtypes)


```
![DF](env/Code/TerminalOutput/DF.png)
![DF data types](env/Code/TerminalOutput/DFdatatypes.png)

As we can see the string data in the data-frame is in the form of object, we need to convert it back to string to work on it.

```python

string_col = df.select_dtypes(include="object").columns
df[string_col]=df[string_col].astype("string")
print(df.dtypes)
```
![DF data types](env/Code/TerminalOutput/DFdatatypes.png)

