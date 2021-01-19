#!/usr/bin/env python
# coding: utf-8

# # Evaluacion y Validacion del Modelo con datos de 2018
# 
# ### En el presente apendice se encuentra el procedimiento ejecutado para la validacion del modelo con los datos del año 2018 con el objetivo de evaluar el desempeño del mismo, este apendice se contiene:
# 
# ## 1. Lectura de las bases de datos y eliminacion de variables altamente correlacionadas.
# ## 2. Modelamiento por StatsModels (OLS Regression)
# ## 3. Modelamiento por Sklearn (LinearRegression).

# # CODIGO
# ## 1. Lectura de las bases de datos y eliminacion de variables altamente correlacionadas.

# In[1]:


# Imports necesarios

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import seaborn as sns

from patsy import dmatrices
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


path_file = "C:/Users/Pipe/Desktop/Apendices_Final/Apendice_G.xlsx"
dta = pd.read_excel(path_file)
df = pd.DataFrame(dta)

path_file_2018 = "C:/Users/Pipe/Desktop/Apendices_Final/Apendice_I.xlsx"
dta_2018 = pd.read_excel(path_file_2018)
df_2018 = pd.DataFrame(dta_2018)


# In[3]:


dta.drop(['PUBLICI','TOTAL_REM2','INVPRO','SERVEXT2','SERVINT'], axis=1, inplace=True)
dta


# In[4]:


dta_2018.drop(['TOTAL_REM','INV_PRO','SERV_EXT','SERV_INT'], axis=1, inplace=True)
dta_2018


# In[5]:


X = pd.DataFrame(dta.iloc[:,0:4].values)
print(X)
X = sm.add_constant(X)
y = dta.iloc[:,[4]]
print(y)


# In[6]:


XX = pd.DataFrame(dta_2018.iloc[:,0:4].values)
print(XX)
XX = sm.add_constant(XX)
yy = dta_2018.iloc[:,[4]]
print(yy)


# # 2. Modelamiento por StatsModels (OLS Regression)

# In[7]:


mod_ols = sm.OLS(y, X)
res_ols = mod_ols.fit()
print(res_ols.summary())

y_pred_o = res_ols.predict(X)
error_o = np.sqrt(mean_squared_error(y, y_pred_o))
print("RESULTADOS DE LA PREDICCION PARA AÑOS 2014-2017 ")
print('Error: ', error_o)
print('R2: ', res_ols.rsquared,"\n")

y_pred_o2 = res_ols.predict(XX)
error_o2 = np.sqrt(mean_squared_error(yy, y_pred_o2))
r2 = r2_score(yy,y_pred_o2)

print("RESULTADOS DE LA PREDICCION PARA EL AÑO 2018 CON EL MODELO DE REGRESION FINAL")
print('Error: ', error_o2)
print('R^2: ',r2)


# # 3. Modelamiento por Sklearn (LinearRegression)

# In[8]:


reg = LinearRegression()
reg = reg.fit(X, y)

y_pred = reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
r2 = reg.score(X, y)

print("El error es: ", error)
print("El valor de r^2 es: ", r2)
print("Los coeficientes son: ", reg.coef_)


# In[9]:


y_pred2 = reg.predict(XX)
error2 = np.sqrt(mean_squared_error(yy, y_pred2))
r22 = reg.score(XX, yy)
print("El error es: ", error2)
print("El valor de r^2 es: ", r22)

