#!/usr/bin/env python
# coding: utf-8

# # Tratamiento de datos y aplicacion de herramientas estadisticas
# 
# ### En el presente apendice se encuentran tratamientos de datos y aplicacion de herramientas estadisticas empleadas para el analisis preliminar al modelo de regresion, los procedimentos contenidos se encuentran en el siguiente orden:
# 
# ## 1. Lectura de bases de datos.
# ## 2. Imputación y Normalización.
# ## 3. Mapa de calor (correlaciones).
# ## 4. Metodo de Codo y Clustering.
# ## 5. Análisis de componentes principales (PCA). 

# # CODIGO
# # 1. Lectura de bases de datos.

# In[1]:


#Importar módulos o paquetes necesarios

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import mglearn
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from scipy import stats


# In[2]:


#Importar datos

d_eac= pd.read_csv(r"C:\Users\Grios\Desktop\pipe\Proyecto\Correcciones\Apendice_B.csv", sep=";", engine="python")
d_eac


# In[3]:


#Retirar variables

d_eac.drop(['IDOJ1','IDAIO','BRUTA','PUBLICI' ], axis=1, inplace=True)


# # 2. Imputacion datos y Normalizacion.

# In[4]:


#Hallar la media de cada variable

p_arr = d_eac["ARRIENDO"].mean()
p_sint=d_eac['SERV_INT'].mean()
p_transp=d_eac['TRANSP'].mean()
p_sext=d_eac['SERV_EXT'].mean()
p_totrem=d_eac['TOTAL_REM'].mean()
p_venta=d_eac['VENTA'].mean()
p_totper=d_eac['TOT_PERSO'].mean()
p_invpro=d_eac['INV_PRO'].mean()
p_cto=d_eac['CTO'].mean()


# In[5]:


#Asignar valores a las celdas vacias

d_eac.ARRIENDO= d_eac.ARRIENDO.replace(np.nan, p_arr)
d_eac.SERV_INT= d_eac.SERV_INT.replace(np.nan, p_sint)
d_eac.TRANSP= d_eac.TRANSP.replace(np.nan, p_transp)
d_eac.SERV_EXT= d_eac.SERV_EXT.replace(np.nan, p_sext)
d_eac.TOTAL_REM= d_eac.TOTAL_REM.replace(np.nan, p_totrem)
d_eac.VENTA= d_eac.VENTA.replace(np.nan, p_venta)
d_eac.TOT_PERSO= d_eac.TOT_PERSO.replace(np.nan, p_totper)
d_eac.INV_PRO= d_eac.INV_PRO.replace(np.nan, p_invpro)
d_eac.CTO= d_eac.CTO.replace(np.nan, p_cto)

datos=pd.DataFrame(d_eac)
datos


# ### Normalización de los Datos

# In[6]:


D_eac=(datos-datos.min())/(datos.max()-datos.min())
D_eac


# In[7]:


D_eac.drop(['AÑO','IDNOREMP','CORRELA'], axis=1, inplace=True)

D_eac.to_excel('C:/Users/Grios/Desktop/pipe/Proyecto/Correcciones/Apendice_D.xlsx')


# # 3. Mapa de Calor (Correlaciones)

# In[8]:


correlation_mat = D_eac.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation_mat, cmap= 'Oranges' ,annot = True)
plt.show()


# # 4. Metodo de Codo y Clustering.
# 
# Búsqueda de la cantidad óptima de clusters
# calculando que tan similares son los individuos dentro de los clusters

# In[9]:


sc = []
for i in range (1,11):                             #crear diferentes agrupaciones con 10 iteraciones
    kmeans= KMeans (n_clusters = i, max_iter=300)
    kmeans.fit(D_eac)
    sc.append (kmeans.inertia_)


# Gráfica de resultados de SC (suma de los cudrados) para formar el Codo de Jambú

# In[10]:


plt.plot(range(1,11), sc)        #eje X toma valores del 1 al 10 y el eje Y los valores de sc
plt.title("Codo de Jambú")
plt.xlabel("Número de Clusters")
plt.ylabel('SC')                 #SC, indicador de que tan similares son los individuos dentro de los clusters
plt.show()


# El punto en donde deja de disminuir de manera drástica es en 4 o en 5

# ### Aplicando el método k-means a los datos

# In[11]:


clustering= KMeans(n_clusters=5, max_iter=300)
clustering.fit(D_eac)


# In[12]:


CORR=datos.iloc[:,[2]]


# ### Agregando la clasificación al archivo original

# In[13]:


D_eac['Cluster']= clustering.labels_

D_eac['CORRELA']=CORR
D_eac


# ### Agrupar datos por clúster y actividad

# In[14]:


data=D_eac.groupby(["Cluster","CORRELA"])[['ARRIENDO','SERV_INT','TRANSP','SERV_EXT','TOTAL_REM','VENTA','TOT_PERSO','INV_PRO','CTO']].count()
data


# In[15]:


# Se seleccionan los datos atípicos
x= (D_eac[(D_eac["Cluster"]==1)].index, D_eac[(D_eac["Cluster"]==2)].index, D_eac[(D_eac["Cluster"]==3)].index)
x


# ### Eliminar valores atípicos

# In[16]:


D_eac=D_eac.drop([742, 1690, 2617, 3394, 3395, 3433, 3466, 507, 574, 857, 1449, 1521, 1806, 2379, 2451, 2729, 3241, 3307, 3573, 26, 572, 588, 665, 667, 809, 982, 1519, 1609, 1611, 1612, 1756, 1930, 2449, 2538, 2540, 2541, 2677, 3305, 3392, 3522],axis=0)
D_eac


# In[17]:


#Diagramas de caja de cada una de las variables

sns.boxplot(x='Cluster', y='ARRIENDO', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
sns.boxplot(x='Cluster', y='SERV_INT', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
sns.boxplot(x='Cluster', y='TRANSP', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
sns.boxplot(x='Cluster', y='SERV_EXT', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
#sns.boxplot(x='Cluster', y='TOTAL_REM', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
#plt.show()
sns.boxplot(x='Cluster', y='VENTA', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
#sns.boxplot(x='Cluster', y='TOT_PERSO', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
#plt.show()
sns.boxplot(x='Cluster', y='INV_PRO', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()
sns.boxplot(x='Cluster', y='CTO', hue='CORRELA', data=D_eac, palette="Set3", linewidth=0.5)
plt.show()


# In[18]:


D_eac.drop(['Cluster','CORRELA'], axis=1, inplace=True)


# # 5. Análisis de componentes principales (PCA).

# In[19]:


# Guardar en 'VENT' la vatiable 'VENTA'

VENT=D_eac.iloc[:,[6]]


# In[20]:


#Eliminar la columna 'VENTA' del dataframe

D_eac.drop(['VENTA'], axis=1, inplace=True)


# In[21]:


#Generar la matriz de covarianza

Cov=np.cov(D_eac.T)


# In[22]:


cov=pd.DataFrame(Cov)
cov.columns=['ARRIENDO','SERV_INT','TRANSP','TOTAL_REM','TOT_PERSO','SERV_EXT','INV_PRO','CTO']
cov.rename(index={0:'ARRIENDO',1:'SERV_INT', 2:'TRANSP',3:'TOTAL_REM', 4:'TOT_PERSO',5:'SERV_EXT',
                  6:'INV_PRO',7:'CTO'}, inplace=True)
cov

                                     #MATRIZ DE COVARIANZA       


# In[23]:


#Agregar nuevamente la variable 'VENTA' al dataframe

D_eac['VENTA']=VENT
D_eac


# In[24]:


# Dividir la matriz del dataset en dos partes

X = pd.DataFrame(D_eac.iloc[:,0:8].values)
# la submatriz X contiene los valores de 8 columnas del dataframe y todas las filas (variables independientes)
print(X)

y = D_eac.iloc[:,8].values
# El vector y contiene los valores de la columna ventas para todas las filas (variable dependiente)
print(y)


# In[25]:


#Guardar el dataframe

#D_eac.to_excel('C:/Users/Grios/Desktop/pipe/Proyecto/Correcciones/archivo_datos.xlsx')


# ### Grafica de sedimentación

# In[26]:


pca=PCA(n_components=6)
pca.fit(X)
X_pca=pca.transform(X) 

print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:2]))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.show()


# ### Varianza explicada

# In[27]:


# Calculamos los autovalores y autovectores de la matriz

autovalores, autovectores = np.linalg.eig(Cov)

print('Eigenvectors \n%s' %autovectores)
print('\nEigenvalues \n%s' %autovalores)


# In[28]:


# A partir de los autovalores, calculamos la varianza explicada

tot = sum(autovalores)
var_exp = [(i / tot)*100 for i in sorted(autovalores, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(8), var_exp, alpha=0.5, align='center', label='Varianza individual explicada', color='g')
    plt.step(range(8), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='center right')
    plt.tight_layout()


# In[29]:


#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(autovalores[i]), autovectores[:,i]) for i in range(len(autovalores))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])

#Esta lista permite hallar el porcentaje de varianza explicada por cada componente


# In[30]:


#Tomar las componenetes que se conservan para el modelo (2)

autovaloresc=autovalores [[0,1]]
autovaloresc


# In[31]:


autovectoresc=autovectores [:,[0,1]]
autovectoresc


# In[32]:


#importar tabulate

from tabulate import tabulate


# In[33]:


#Correlacion entre variables y componentes

rxy =[]
for i in range(8):
    for k in range(2):
        aki=autovectoresc[i,k]
        lk=autovaloresc[k]
        Cii=Cov[i,i]
        rxiyk=aki*np.sqrt(lk)/np.sqrt(Cii)
        rxy.append(rxiyk)

print("Correlación entre las variables originales" '\n' "y los componenetes extraídos: ", '\n', '\n',
     tabulate(np.array(rxy).reshape(8,2), ["Comp 1","Comp 2"],
             showindex=['ARRIENDO','SERV_INT','TRANSP','SERV_EXT','TOTAL_REM','TOT_PERSO','INV_PRO','CTO']))


# In[34]:


comp = X_pca[:,:2]
trans_comp = np.transpose(comp)
trans_comp

import xlsxwriter

workbook = xlsxwriter.Workbook('C:/Users/Grios/Desktop/pipe/Proyecto/Correcciones/componentes.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(trans_comp):
    worksheet.write_column(row, col, data)

workbook.close()


# In[ ]:




