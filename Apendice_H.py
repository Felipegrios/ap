#!/usr/bin/env python
# coding: utf-8

# # Modelos de Regresion
# 
# ### Se presentan la totalidad de los modelos elaborados para el analisis estadistico y sus respectivos resultados, 2 pares de modelos estadisticos por cada matriz de datos, donde en la primer matriz se encuentran las variables originales y en la segunda los componentes principales, los modelos y procedimientos se distribuyen de la siguiente manera:
# 
# ## 1. Lectura de bases de datos.
# ## 2. Pruebas de hipotesis para significancia en correlaciones entre pares de variables.
# ## 3. Modelamiento por StatsModels (OLS Regression) variables no correlacionadas.
# ## 4. Modelamiento por Sklearn (LinearRegression) variables no correlacionadas.
# ## 5. Modelamiento por StatsModels (OLS Regression) componentes principales.
# ## 6. Modelamiento por Sklearn (LinearRegression) componentes principales.
# 

# # CODIGO
# # 1. Lectura de bases de datos.

# In[1]:


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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


# In[2]:


path_file = "C:/Users/Grios/Desktop/pipe/Proyecto/Correcciones/Apendice_G.xlsx"
dta = pd.read_excel(path_file)
df = pd.DataFrame(dta)
df.info()


# # 2. Pruebas de hipotesis para correlaciones entre pares de variables.

# ### Correlacion de spearman

# In[3]:


dta.corr('spearman')


# ### Pruebas de hipotesis

# In[4]:


#SPEARMAN
Fil = dta['PUBLICI']
Col = dta['TRANSP']

coef, p = spearmanr(Fil, Col)
print('Coeficiente de correlacion de Spearman: %.3f' % coef)

#Interpretar significancia
alpha = 0.05  #95% de probalidad de acierto de que las pruebas no esten correlacionadas
if p > alpha:
    print('Las pruebas estan no relacionadas (fallo en rechazar la hipotesis) p=%.3f' % p)
else:
    print('Las pruebas estan relacionadas (rechaza la hipotesis) p=%.3f' % p)


# In[5]:


#SPEARMAN
Fil = dta['TOTAL_REM2']
Col = dta['TOTPERSO']

coef, p = spearmanr(Fil, Col)
print('Coeficiente de correlacion de Spearman: %.3f' % coef)

#Interpretar significancia
alpha = 0.05  #95% de probalidad de acierto de que las pruebas no esten correlacionadas
if p > alpha:
    print('Las pruebas estan no relacionadas (fallo en rechazar la hipotesis) p=%.3f' % p)
else:
    print('Las pruebas estan relacionadas (rechaza la hipotesis) p=%.3f' % p)


# ## Eliminacion de variables invalidas.

# In[6]:


dta.drop(['PUBLICI','TOTAL_REM2','INVPRO','SERVEXT2'], axis=1, inplace=True)
dta


# In[7]:


X = pd.DataFrame(dta.iloc[:,0:5].values)
print(X)
X = sm.add_constant(X)
y = dta.iloc[:,[5]]
print(y)


# # 3. Modelamiento por StatsModels (OLS Regression) variables no correlacionadas.
# 

# In[8]:


mod_ols = sm.OLS(y, X)
res_ols = mod_ols.fit()
print(res_ols.summary())

y_pred_o = res_ols.predict(X)
error_o = np.sqrt(mean_squared_error(y, y_pred_o))
print('Error: ', error_o)
print('R2: ', res_ols.rsquared)


# # 4. Modelamiento por Sklearn (LinearRegression) variables no correlacionadas.
# 

# In[9]:


reg = LinearRegression()
regR = reg.fit(X, y)

y_pred = regR.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
r2 = reg.score(X, y)


# In[10]:


print("El error es: ", error)
print("El valor de r^2 es: ", r2)
print("Los coeficientes son: ", reg.coef_)


# ## Lectura de matriz de datos componentes principales.
# 

# In[11]:


path_file = "C:/Users/Grios/Desktop/pipe/Proyecto/Correcciones/Apendice_F.xlsx"
dta_co = pd.read_excel(path_file)
df_co = pd.DataFrame(dta_co)
df_co.info()


# In[12]:


dta_co


# In[13]:


X = pd.DataFrame(dta_co.iloc[:,0:2].values)
print(X)
X = sm.add_constant(X)
y = dta_co.iloc[:,[2]]
print(y)


# # 5. Modelamiento por StatsModels (OLS Regression) componentes principales.

# In[14]:


mod_ols = sm.OLS(y, X)
res_ols = mod_ols.fit()
print(res_ols.summary())

y_pred_o = res_ols.predict(X)
error_o = np.sqrt(mean_squared_error(y, y_pred_o))
print('Error: ', error_o)
print('R2: ', res_ols.rsquared)


# # 6. Modelamiento por Sklearn (LinearRegression) componentes principales.
# 

# In[15]:


reg = LinearRegression()
reg = reg.fit(X, y)

y_pred = reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
r2 = reg.score(X, y)


# In[16]:


print("El error es: ", error)
print("El valor de r^2 es: ", r2)
print("Los coeficientes son: ", reg.coef_)

