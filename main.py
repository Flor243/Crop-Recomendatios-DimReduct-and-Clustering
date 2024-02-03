# -*- coding: utf-8 -*-
"""
# Trabajo Práctico 1

Minería de datos

Integrantes:
- Fernández, Florencia
- Salvañá, Leandro


"""
# Se comienza por importar las liberías necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats import chi2
from sklearn import decomposition
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.covariance import MinCovDet
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from gap_statistic import OptimalK
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage


"""  ----------------------------------
     ----------------------------------
      2 - Análisis Exploratorio (EDA)
     ----------------------------------
     ----------------------------------
"""

crop_recommendation_df = pd.read_csv('Crop_recommendation.csv')
crop_recommendation_df

# Se averiguan la cantidad de filas y columnas en el dataset
crop_recommendation_df.shape

"""El dataset posee 2200 filas y 8 columnas correspondientes a variables que describen los datos"""

# Se comprueba si existen o no filas duplicadas en el dataset
crop_recommendation_df.duplicated().sum()

"""El dataset no posee filas duplicadas. Es decir, no existe información redundante.

### 2.2 - Conocer las columnas
En este paso se imprime una lista de las columnas con sus respectivos tipos y cantidad de datos.
"""

crop_recommendation_df.info()

# Se comprueba si existen valores nulos en las columnas del dataset
crop_recommendation_df.isna().sum()

"""Se observa que no hay presencia de valores faltantes y que el tipo de dato especificado para cada columna se corresponde
 con lo que representan. Es decir, las columns N, P y K se corresponden con el tipo de dato numérico discreto, temperature, 
 humidity, ph y rainfall con un valor numérico con decimales y label con texto.

Por lo divisado, no será requerido completar valores faltantes en el dataset ni realizar un 
cambio en el tipo de dato de las columnas del mismo.
"""

# Se quiere conocer qué cultivos fueron evluados para formar este dataset
unique_labels = crop_recommendation_df['label'].unique()
print("Valores únicos en la columna 'label':")
count = 0
for label in unique_labels:
    print(f'- {label}')
    count += 1
print('\n')
print(f'Cantidad de valores únicos en la columna label: {count}')

# Se observa la cantidad de filas presentes para cada cultivo
crop_recommendation_df.label.value_counts()

"""Existe la misma cantidad de registros para cada alimento.

Se analizará si existen valores en las columnas que se encuentren fuera de rango en relación con la característica que representan.
"""

# Se comprueba si existen valores negativos para las columnas de potasio (K), fósforo (P) y nitrógeno (N)
crop_recommendation_df.loc[(crop_recommendation_df['K'] < 0) | (crop_recommendation_df['P'] < 0) | (crop_recommendation_df['N'] < 0)]

"""Considerando que las unidades para las columnas K, P y N son mg/kg, no podría representarse con valores negativos.

No se observa presencia de registros con valores negativos para dichas columnas.
"""

# Se comprueba si existen valores negativos para las mediciones de los milímetros de lluvia
crop_recommendation_df.loc[(crop_recommendation_df['rainfall'] < 0.0)]

"""Considerando que la unidad para la columna rainfall es mm no podría representarse con valores negativos. 
Es decir, no hay mediciones de mm de lluvia negativos.

No se observa presencia de registros con valores negativos para dicha columna.
"""

# Se observa si existen valores de porcentaje menores a 0 o mayores a 100 para la columna de humedad (humidity)
crop_recommendation_df.loc[(crop_recommendation_df['humidity'] < 0.0) | (crop_recommendation_df['humidity'] > 100.0)]

"""Considerando que la unidad para la columna humidity es % (porcentaje) no podría representarse con valores menores a cero o mayores a 100.

No se observa presencia de registros con valores inadecuados para dicha columna.
"""

# Se observa si existen valores menores a 0 o mayores a 14 para la columna ph
crop_recommendation_df.loc[(crop_recommendation_df['ph'] < 0.0) | (crop_recommendation_df['ph'] > 14.0)]

"""Considerando que los valores que puede tener el ph se encuentran entre 0 y 14 incluidos, 
no podría representarse con valores menores a cero o mayores a 14.

No se observa presencia de registros con valores inadecuados para dicha columna.

### 2.3 - Medidas estadísticas y de localización
En este paso se estudian, para cada columna, medidas de localización como mínimo, máximo, cuartiles, y de centralidad como la mediana y la media.

 Estas mediciones proporcionan una visión resumida de la distribución de los datos. Esto puede aportar valiosa 
 nformación para el análisis e interpretación de los datos de suelo y su relación con los alimentos.
"""

# Se obtienen medidas estadísticas para el dataset completo
crop_recommendation_df.describe()

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

"""
### 2.5 - Visualización de gráficos y matriz de correlación
"""

# Columnas
columns = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
]

# Colores para los boxplots
colors = sns.color_palette('husl', n_colors=len(columns))

# Crear subplots
fig, axes = plt.subplots(len(columns), 1, figsize=(8, 10), sharex=False)

# Generar boxplots horizontales para cada columna
for i, col in enumerate(columns):
    sns.boxplot(data=crop_recommendation_df, x=col, ax=axes[i], color=colors[i], orient='h')
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_xlabel('')  # Elimina la etiqueta del eje x para mayor claridad

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar los gráficos
plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

# Define la paleta de colores
colors = sns.color_palette('husl', n_colors=len(crop_recommendation_df.columns) - 1)  # Excluye la columna 'label'

# Filtra las columnas que no sean 'label'
numeric_columns = [column for column in crop_recommendation_df.columns if column != 'label']

# Configura el tamaño de la figura
plt.figure(figsize=(12, 8))

# Itera a través de cada columna numérica y crea un histograma (excluyendo 'label')
for i, column in enumerate(numeric_columns):
    plt.subplot((len(numeric_columns) + 1)//2, 2, i+1)  # Divide la figura en subplots
    sns.histplot(crop_recommendation_df[column], color=colors[i], kde=True)  # Crea el histograma
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')

# Ajusta el espacio entre subplots
plt.tight_layout()

# Muestra los gráficos
plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""


# Calcular la matriz de correlación
correlation_matrix = crop_recommendation_df[['K', 'P', 'N', 'temperature', 'humidity', 'ph', 'rainfall']].corr()

# Crear un mapa de calor (correlograma)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='rocket', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlograma de Todas las Columnas')
plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""


sns.pairplot(crop_recommendation_df[['K', 'P', 'N', 'temperature', 'humidity', 'ph', 'rainfall']], plot_kws = {'color': '#08415c', 'marker': 'p'},
             diag_kws = {'color': '#ff5400'}, corner = True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

# Imprime la cantidad de datos antes de eliminar los outliers
print("Cantidad de datos antes de eliminar outliers:", len(crop_recommendation_df))

# Define un nivel de significancia (puedes ajustarlo según tus necesidades)
alpha = 0.01

# Guarda los índices del DataFrame original
original_indices = crop_recommendation_df.index

# Calcula la matriz de covarianza robusta utilizando el Minimum Covariance Determinant
mcd = MinCovDet().fit(crop_recommendation_df.drop(columns=['label']))

# Calcula la matriz de inversa de la covarianza robusta
cov_inv = np.linalg.inv(mcd.covariance_)

# Calcula la distancia de Mahalanobis para cada punto en el DataFrame original
mahalanobis_dist = np.zeros(len(crop_recommendation_df))
for i, row in enumerate(crop_recommendation_df.drop(columns=['label']).values):
    mahalanobis_dist[i] = np.sqrt(np.dot(np.dot((row - mcd.location_).T, cov_inv), (row - mcd.location_)))

# Calcula el valor crítico de la distancia de Mahalanobis
df = len(crop_recommendation_df.columns) - 1
threshold = chi2.ppf(1 - alpha, df)

# Filtra los datos que no son outliers y mantiene las etiquetas de clase
crop_recommendation_df_filtered = crop_recommendation_df.loc[mahalanobis_dist < threshold]

# Imprime la cantidad de datos después de eliminar los outliers
print("Cantidad de datos después de eliminar outliers:", len(crop_recommendation_df_filtered))

crop_recommendation_df_filtered

"""Se quiere conocer qué valores fueron eliminados para comprobar que no corresponden todos a la misma clase.

Si esto fuera así, se estaría eliminando gran parte de los datos que corresponden a un único grupo. El tener menos datos puede llevar a 
perder representatividad y aumentar el sesgo.

Queremos conocer si aquellos registros que se consideran outliers corresponden a una única clase ya que de ser así, tendríamos que considerarlos para el análisis.
"""

# Encontrar filas diferentes
diferentes = pd.concat([crop_recommendation_df, crop_recommendation_df_filtered]).drop_duplicates(keep=False)

# El DataFrame "diferentes" ahora contiene las filas que son diferentes entre los dos DataFrames
diferentes

"""Considerando que casi la totalidad de los datos eliminados en la consideración de outliers son grapes se procederá a conocer 
si efectivamente corresponden a valores outliers dentro de esta clase específica o si los datos eliminados integran el conjunto de observaciones a analizar.

Para ello se realizará un boxplot únicamente para la clase grapes.
"""

# Columnas
columns = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
]

# Colores para los boxplots
colors = sns.color_palette('husl', n_colors=len(columns))

# Crear subplots
fig, axes = plt.subplots(len(columns), 1, figsize=(8, 10), sharex=False)

# Generar boxplots horizontales para cada columna en la clase grapes
for i, col in enumerate(columns):
    sns.boxplot(data=crop_recommendation_df[crop_recommendation_df['label'] == 'grapes'], x=col, ax=axes[i], color=colors[i], orient='h')
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_xlabel('')  # Elimina la etiqueta del eje x para mayor claridad

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar los gráficos
plt.show(block=True)

# Cantidad total de filas con el label grape en el dataset original
total_grapes = crop_recommendation_df[crop_recommendation_df['label'] == 'grapes']['label'].count()

# Cantidad de filas con el label grapes eliminadas como outliers
deleted_grapes = diferentes[diferentes['label'] == 'grapes']['label'].count()

print(f'Cantidad total de filas con el label grape en el dataset original: {total_grapes}')
print(f'Cantidad de filas con el label grapes eliminadas como outliers: {deleted_grapes}')

"""Se repiten las observaciones para la clase apple, para la que se habían eliminado dos registros"""

# Columnas
columns = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
]

# Colores para los boxplots
colors = sns.color_palette('husl', n_colors=len(columns))

# Crear subplots
fig, axes = plt.subplots(len(columns), 1, figsize=(8, 10), sharex=False)

# Generar boxplots horizontales para cada columna en la clase grapes
for i, col in enumerate(columns):
    sns.boxplot(data=crop_recommendation_df[crop_recommendation_df['label'] == 'apple'], x=col, ax=axes[i], color=colors[i], orient='h')
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_xlabel('')  # Elimina la etiqueta del eje x para mayor claridad

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar los gráficos
plt.show(block=True)

# Cantidad total de filas con el label grape en el dataset original
total_apple = crop_recommendation_df[crop_recommendation_df['label'] == 'apple']['label'].count()

# Cantidad de filas con el label apple eliminadas como outliers
deleted_apple = diferentes[diferentes['label'] == 'apple']['label'].count()

print(f'Cantidad total de filas con el label grape en el dataset original: {total_apple}')
print(f'Cantidad de filas con el label apple eliminadas como outliers: {deleted_apple}')

"""En resumen, no se eliminarán datos ya que los mismos representan mayormente a un cultivo y eliminarlos podría quitar representatividad.

Como lo que se tiene en el dataset proporcionado es, en definitiva, información sobre distintos tipos de cultivos, se procederá a explorar las distintas variables por cada cultivo representado en los datos.
"""

#Se define la paleta de colores
colores_personalizados =  ["#F72585",  "#b5179e",  "#7209b7", "#560bad", "#3a0ca3", "#3f37c9", "#4361ee", "#4cc9f0",
                           "#007f5f",  "#2b9348",  "#80b918", "#d4d700", "#eeef20", "#ff7b00", "#312244",
                           "#2a6f97", "#b69121", "#1a7431", "#4ad66d", "#c55df6", "#ff924c", "#003d5c"]

for i, variable in enumerate(crop_recommendation_df.select_dtypes(include=['int', 'float']).columns.tolist()):
    plt.subplots(figsize=(15,8))
    sns.boxplot(x=variable, y='label', data=crop_recommendation_df, palette = colores_personalizados)

    plt.title(f'Diagramas de caja de la variable {str(variable)} para cada cultivo', fontsize = 18)
    plt.xlabel(f'Valores de {str(variable)}', fontsize = 14)
    plt.ylabel('Nombre del cultivo', fontsize = 18)
    plt.show(block=True)

# Creación del dataframe con las medidas estadísticas por cultivo
columnas = crop_recommendation_df.columns.to_list()
columnas.append('estadistica')
estadisticas_por_cultivo_df = pd.DataFrame(columns=columnas)


  # Estadísticas
for cultivo in list(crop_recommendation_df.label.unique()):
  nueva_fila_media = []
  nueva_fila_mediana = []
  nueva_fila_desvio_estandar = []
  for columna in columnas[:len(columnas)-2]:
    # Crear filas como lista de valores para todas las columnas de ése cultivo
    media = crop_recommendation_df[crop_recommendation_df['label'] == cultivo][columna].mean()
    nueva_fila_media.append(media)
    mediana = crop_recommendation_df[crop_recommendation_df['label'] == cultivo][columna].median()
    nueva_fila_mediana.append(mediana)
    desviacion_estandar = crop_recommendation_df[crop_recommendation_df['label'] == cultivo][columna].std()
    nueva_fila_desvio_estandar.append(desviacion_estandar)
  
  # Agregar la fila al DataFrame
  nueva_fila_media.append(cultivo)
  nueva_fila_media.append('media')
  
  estadisticas_por_cultivo_df.loc[len(estadisticas_por_cultivo_df)] = nueva_fila_media
  nueva_fila_mediana.append(cultivo)
  nueva_fila_mediana.append('mediana')
  estadisticas_por_cultivo_df.loc[len(estadisticas_por_cultivo_df)] = nueva_fila_mediana
  nueva_fila_desvio_estandar.append(cultivo)
  nueva_fila_desvio_estandar.append('desvio_estandar')
  estadisticas_por_cultivo_df.loc[len(estadisticas_por_cultivo_df)] = nueva_fila_desvio_estandar

# Pivotar la tabla
estadisticas_por_cultivo_df = estadisticas_por_cultivo_df.pivot_table(index='label', columns='estadistica', values=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], aggfunc='first')

# Resetear el índice
estadisticas_por_cultivo_df = estadisticas_por_cultivo_df.reset_index()
# Verificar el DataFrame actualizado
print(estadisticas_por_cultivo_df)

# Iterar sobre las columnas y crear gráficos de barras
for columna in crop_recommendation_df.select_dtypes(include=['int', 'float']).columns.tolist():
    plt.figure(figsize=(15, 8))
    for i, cultivo in enumerate(estadisticas_por_cultivo_df['label']):
        plt.bar(cultivo, estadisticas_por_cultivo_df[columna]['media'][i], color=colores_personalizados[i], label=cultivo)
        plt.axhline(y=crop_recommendation_df[columna].mean(), color='r', linestyle='--')

    plt.xlabel('Cultivo')
    plt.ylabel(columna)
    plt.title(f'{columna} por Cultivo')
    plt.xticks(rotation=45)
    plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

for columna in crop_recommendation_df.select_dtypes(include=['int', 'float']).columns.tolist():
  superan_la_media = 'Los cultivos que superan la media de ' + columna + ' son: '
  no_superan_la_media = 'Los cultivos que no superan la media de ' + columna + ' son: '

  for i, cultivo in enumerate(estadisticas_por_cultivo_df['label']):
    media_cultivo = estadisticas_por_cultivo_df[columna]['media'][i]
    if media_cultivo > crop_recommendation_df[columna].mean():
      superan_la_media += f' {cultivo} '
    else:
      no_superan_la_media += f' {cultivo} '

  print(superan_la_media)
  print(no_superan_la_media)

  print('\n')

"""Se hará un gráfico de barras solo para los nutrientes puesto que se observa una distribución similar en ellos"""

# Filtrar el DataFrame solo para las columnas K, P y N
df_filtrado = estadisticas_por_cultivo_df.loc[:, ['label', 'K', 'P', 'N']]
cultivos = df_filtrado['label'].to_list()
N = 22
ind = np.arange(N)
width = 0.25
plt.figure(figsize=(15, 8))
K_vals = df_filtrado['K']['media'].to_list()
bar1 = plt.bar(ind, K_vals, width, color = 'r')

N_vals = df_filtrado['N']['media'].to_list()
bar2 = plt.bar(ind+width, N_vals, width, color='g')

P_vals = df_filtrado['P']['media'].to_list()
bar3 = plt.bar(ind+width*2, P_vals, width, color = 'b')

plt.xlabel("Cultivos")
plt.ylabel('Medias')
plt.title("Medias de cada cultivo para los nutrientes del suelo K, P y N")

plt.xticks(ind+width,cultivos, rotation = 45)
plt.legend( (bar1, bar2, bar3), ('K', 'N', 'P') )
plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

fig = px.scatter_3d(crop_recommendation_df, x='N', y='P', z='K',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'label'})
fig.update_layout(title='Distribución de cultivos para las variables K, P y N')
fig.show(block=True)

fig = px.scatter_3d(crop_recommendation_df, x='rainfall', y='humidity', z='ph',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'label'})
fig.update_layout(title='Distribución de cultivos para las variables rainfall humidity y ph')
fig.show(block=True)

fig = px.scatter_3d(crop_recommendation_df, x='temperature', y='ph', z='N',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'label'})
fig.update_layout(title='Distribución de cultivos para las variables temperature, ph y N')
fig.show(block=True)

fig = px.scatter_3d(crop_recommendation_df, x='temperature', y='rainfall', z='N',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'label'})
fig.update_layout(title='Distribución de cultivos para las variables temperature, rainfall y N')
fig.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 2 - Análisis Exploratorio (EDA)"""
"""************************************************************************************************"""

# Excluir la columna "label" antes de estandarizar los datos
data_to_standardize = crop_recommendation_df.drop(columns=['label'])

# Estandarizar los datos
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_to_standardize)



"""  ----------------------------------
     ----------------------------------
                 3 - PCA
     ----------------------------------
     ----------------------------------
"""
"""Realizar PCA y determinar el número de componentes principales considerando alguno de los 3 criterios dados en la práctica. Graficar la varianza acumulada y las componentes de PCA en un grafico 2 o 3D con sus respectivas clases.
"""

# Realizar el PCA en los datos estandarizados
pca = PCA(n_components=data_to_standardize.shape[1])
principal_components = pca.fit_transform(standardized_data)

# Columns for PCA dataframe
pc_columns = [f'PC{i}' for i in range(1, len(data_to_standardize.columns) + 1)]

# PC dataframe
pca_df = pd.DataFrame(
    data=principal_components,
    columns=pc_columns)
pca_df['label'] = crop_recommendation_df['label']

pca_df

"""En el anterior dataframe vemos la proyección de cada registro en cada una de las nuevas variables.

A continuación, se observarán en forma de tabla y gráficamente las representaciones de los eigenvalues para cada componente.
"""

# Variabilidad explicada por cada componente principal
explained_variance_ratio = pca.explained_variance_ratio_

# Calcular la proporción acumulada de varianza explicada
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Crear un DataFrame con las columnas requeridas
pca_exp_var_df = pd.DataFrame({
    "Eigenvalues": principal_components.var(axis=0),  # Varianza explicada por cada componente principal
    "Proporción de varianza explicada": explained_variance_ratio,
    "Proporción acumulada de varianza explicada": cumulative_variance_ratio
})

pca_exp_var_df

# Calcula los eigenvalues de las componentes principales
eigenvalues = np.var(principal_components, axis=0)

# Crea un DataFrame con los eigenvalues
eigenvalues_df = pd.DataFrame({'Componente Principal': range(1, len(eigenvalues) + 1),
                               'Eigenvalues': eigenvalues})

# Gráfica los eigenvalues
plt.figure(figsize=(10, 6))
plt.bar(eigenvalues_df['Componente Principal'], eigenvalues_df['Eigenvalues'], color='#004733', alpha=0.7)
plt.xlabel('Componente Principal')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues de las Componentes Principales')
plt.grid(True)

# Muestra el DataFrame
print(eigenvalues_df)

# Muestra el gráfico
plt.show(block=True)

"""Vemos que las primeras 4 componentes tienen eigenvalues mayores a uno, indicando una mayor contribución
a la explicación de la varianza en los datos.

A continuación, se visualizará gráficamente la varianza explicada por cada componente y la acumulada.
"""

# Extraer los datos necesarios del DataFrame
component_numbers = range(1, len(pca_exp_var_df) + 1)
explained_variance = pca_exp_var_df['Proporción de varianza explicada']

# Crear el gráfico de líneas para la Proporción de varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(component_numbers, explained_variance, marker='o', color='#da1e37', linestyle='-', linewidth=2)
plt.xlabel('Componentes Principales')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Proporción de Varianza Explicada por Componente Principal')
plt.grid(True)

# Mostrar el gráfico
plt.show(block=True)

"""Al observar el gráfico, puede verse que a partir de la quinta componente la propporción de varianza explicada decae
 considerablemente en relación a las primeras cuatro componentes."""

# Extraer los datos necesarios del DataFrame
component_numbers = range(1, len(pca_exp_var_df) + 1)
explained_variance = pca_exp_var_df['Proporción de varianza explicada']
cumulative_variance = pca_exp_var_df['Proporción acumulada de varianza explicada']

# Crear el gráfico de barras para la Proporción de varianza explicada
plt.figure(figsize=(10, 6))
plt.bar(component_numbers, explained_variance, alpha=0.7, align='center', label='Proporción de Varianza Explicada', color = '#27a300')

# Crear el gráfico de línea para la Proporción acumulada de varianza explicada
plt.plot(component_numbers, cumulative_variance, marker='o', color='#dc5713', label='Proporción Acumulada de Varianza Explicada')

# Etiquetas y título del gráfico
plt.xlabel('Número de Componente Principal')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Proporción de Varianza Explicada y Proporción Acumulada de Varianza Explicada')
plt.legend(loc='upper left')

# Mostrar el gráfico
plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 3 - PCA"""
"""************************************************************************************************"""

# Crea un DataFrame con las tres primeras componentes principales
correlogram_data = pd.DataFrame(data=principal_components[:, :4], columns=['Componente 1', 'Componente 2', 'Componente 3', 'Componente 4'])

# Calcula la matriz de correlación entre las componentes principales
correlation_matrix = correlogram_data.corr()

# Crea un mapa de calor (correlograma) de la matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='rocket', fmt=".2f", linewidths=0.5)
plt.title('Correlograma de las tres primeras Componentes Principales')
plt.show(block=True)

"""Como es de esperar, la correlación entre las componentes principales es nula, es decir que las mismas son ortogonales.

Dado que solo es posible graficar en dos dimensiones o tres, se seleccionarán las primeras 2 y 3 
componentes (ya que son las que tienen las proporciones más elevadas de todas) para realizar los mismos y 
observar visualmente los resultados de la distribución de los datos en el nuevo espacio de dimensiones.
"""

features = crop_recommendation_df.drop(columns=['label']).columns.to_list()

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(principal_components, x=0, y=1, color = pca_df["label"],color_discrete_sequence=colores_personalizados,  labels={'color': 'label'})
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0]*4.5,
        y1=loadings[i, 1]*4.5
    )
    fig.add_annotation(
        x=loadings[i, 0]*5.5,
        y=loadings[i, 1]*5.5,
        ax=0, ay=0,
        xanchor='center',
        yanchor='bottom',
        text=feature,
    )
fig.update_layout(title = "Distribución de cultivo en 2 componentes",width = 1200,height = 600)
fig.show(block=True)
fig = px.scatter_3d(principal_components, x=0, y=1, z=2,
              color=pca_df["label"],color_discrete_sequence=colores_personalizados,  labels={'color': 'label'})
fig.update_layout(title = "Distribución de cultivo en 3 componentes")
fig.show(block=True)

""""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 3 - PCA"""
"""************************************************************************************************"""

# Se obtienen los valores de las componentes
componentes = pca.components_

# Crea un DataFrame para visualizarlos
df_componentes = pd.DataFrame(componentes, columns=crop_recommendation_df.columns.to_list()[0:7])

# Se viualizan los valores de las componentes
print(df_componentes)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 3 - PCA"""
"""************************************************************************************************"""




"""  ----------------------------------
     ----------------------------------
                4 - Isomap
     ----------------------------------
     ----------------------------------
"""
"""
Aplicar Isomap y analizar los resultados obtenidos variando el numero de vecinos y componentes. Realizar un grafico en 2D de utilizando dos componentes.
"""

# Número de componentes principales y vecinos a probar
""" n_components_list = [2, 3, 4]
n_neighbors_list = [50, 100, 150, 200, 250, 300] """
n_components_list = [2]
n_neighbors_list = [10, 20]
# Iterar sobre las combinaciones
for n_neighbors in n_neighbors_list:
    for n_components in n_components_list:
        # Aplicar ISOMAP
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        isomap_result = isomap.fit_transform(standardized_data)

        # Crear un scatter plot en 2D
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=isomap_result[:, 0], y=isomap_result[:, 1], hue=crop_recommendation_df.label, palette=colores_personalizados)
        plt.title(f'ISOMAP - Vecinos: {n_neighbors}, Componentes: {n_components}')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.legend(title='Cultivo', loc='upper right')
        plt.show(block=True)

"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 4 - Isomap"""
"""************************************************************************************************"""

"""
A continuación, se harán gráficos tridimensionales los resultados de ISOMAP para la cantidad 
de vecinos detectada como punto de inflexión en las observaciones anteriores (200), como así también 
la cantidad utilizada anterior a ella y la posterior. De esta manera se quiere tener un 
acercamiento más detallado del cambio producido.
"""
isomap_df = Isomap(n_neighbors=150, n_components=3)
isomap_df.fit(standardized_data)
projections_isomap = isomap_df.transform(standardized_data)

fig = px.scatter_3d(
    projections_isomap, x=0, y=1, z=2,
    color=crop_recommendation_df['label'], color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'}
)
fig.update_traces(marker_size=8)
fig.update_layout(title = "Distribución de cultivos en 3 dimensiones - ISOMAP, 150 vecinos")
fig.show(block=True)

isomap_df = Isomap(n_neighbors=200, n_components=3)
isomap_df.fit(standardized_data)
projections_isomap = isomap_df.transform(standardized_data)

fig = px.scatter_3d(
    projections_isomap, x=0, y=1, z=2,
    color=crop_recommendation_df['label'], color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'}
)
fig.update_traces(marker_size=8)
fig.update_layout(title = "Distribución de cultivos en 3 dimensiones - ISOMAP, 200 vecinos")
fig.show(block=True)

isomap_df = Isomap(n_neighbors=300, n_components=3)
isomap_df.fit(standardized_data)
projections_isomap = isomap_df.transform(standardized_data)

fig = px.scatter_3d(
    projections_isomap, x=0, y=1, z=2,
    color=crop_recommendation_df['label'], color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'}
)
fig.update_traces(marker_size=8)
fig.update_layout(title = "Distribución de cultivos en 3 dimensiones - ISOMAP, 300 vecinos")
fig.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 4 - Isomap"""
"""************************************************************************************************"""




"""  ----------------------------------
     ----------------------------------
                5 - t-SNE
     ----------------------------------
     ----------------------------------
"""
"""
Aplicar t-SNE y analizar los resultados obtenidos variando el número de iteraciones, componentes y perplejidad. Realizar un gráfico en 2D de utilizando dos componentes.
"""

# Añadir la columna 'label' a los datos estandarizados
standardized_df = pd.DataFrame(standardized_data, columns=data_to_standardize.columns)
standardized_df['label'] = crop_recommendation_df['label']

# Configuraciones para t-SNE
""" iterations_list = [500, 1000, 2000]  # Diferentes números de iteraciones
components_list = [2, 3]  # Diferentes números de componentes t-SNE
perplexity_list = [5, 15, 25, 30, 35, 40, 45]  # Diferentes valores de perplejidad """
iterations_list = [250]  # Diferentes números de iteraciones
components_list = [2]  # Diferentes números de componentes t-SNE
perplexity_list = [5, 15]  # Diferentes valores de perplejidad

# Generar gráficos interactivos para cada combinación de parámetros
for iterations in iterations_list:
    for components in components_list:
        for perplexity in perplexity_list:
            # Aplicar t-SNE
            tsne = TSNE(n_components=components, n_iter=iterations, perplexity=perplexity, random_state=42)
            tsne_results = tsne.fit_transform(standardized_data)

            # Crear un DataFrame para los resultados de t-SNE en 2D
            tsne_df = pd.DataFrame(data=tsne_results[:, :2], columns=['Component 1', 'Component 2'])

            # Agregar la columna 'label' nuevamente
            tsne_df['label'] = crop_recommendation_df['label']

            # Crear un gráfico interactivo de dispersión 2D con etiquetas al hacer hover
            fig = px.scatter(tsne_df, x='Component 1', y='Component 2', color='label', color_discrete_sequence=colores_personalizados,
                             labels={'label': 'Etiqueta'}, hover_name='label',
                             title=f't-SNE (Iter: {iterations}, Comp: {components}, Perp: {perplexity})')

            # Ajustar la relación de aspecto para que no estén alargados
            fig.update_layout(autosize=False, width=800, height=600)

            # Mostrar el gráfico interactivo
            fig.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 5 - t-SNE"""
"""************************************************************************************************"""




"""  ----------------------------------
     ----------------------------------
                6 - K-Means
     ----------------------------------
     ----------------------------------
"""
"""
Aplicar K-means y analizar los resultados obtenidos variando el número de clusters y obtener 
el número óptimo de clusters mediante GAP. Realizar un gráfico en 3D de utilizando tres atributos de los datos 
y donde los colores estén asociados a los clusters.

Se aplica la técnica del codo para elegir el número óptimo de clusters.
"""

# Técnica del codo para elegir el número de clusters

distancias = []
for k in range(2, 16):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(standardized_data)
    distancias.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 16), distancias, marker='o', color = '#da1e37')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Distancias')
plt.xticks(np.arange(2, 16))
plt.grid(True)
plt.show(block=True)

"""Se observa en el gráfico que a partir de 7 clusters, las distancias no disminuyen sinificativamente. 
Por lo cual, se elige dicho número para utilizar en el modelo KMeans."""

kmeans = KMeans(n_clusters=7)
kmeans.fit(standardized_data) #Entrenamos el modelo

# El metodo labels_ nos da a que cluster corresponde cada observacion
crop_recommendation_kmeans_df =crop_recommendation_df
crop_recommendation_kmeans_df['Cluster KMeans'] = kmeans.labels_
crop_recommendation_kmeans_df.head()

"""Se graficarán los clusters obtenidos y debajo otro gráfico 3D mostrando los cultivos correspondientes"""

crop_recommendation_kmeans_df['Cluster KMeans'] = crop_recommendation_kmeans_df['Cluster KMeans'].astype(str)

# Diccionario de asignación de colores
colores_personalizados_7 = ['#264653',
    '#2a9d8f',
    '#f4a261',
    '#e76f51',
    '#c1121f',
    '#5f0f40',
    '#a7c957']


# Se grafican los clusters
fig = px.scatter_3d(crop_recommendation_kmeans_df[['K', 'N', 'P']], x='K', y='N', z='P',
              color=crop_recommendation_kmeans_df['Cluster KMeans'], color_discrete_sequence=colores_personalizados_7, labels={'color': 'Cluster KMeans'})
fig.update_layout(title = "7 Clusters de Kmeans para las variables K, P y N")
fig.show(block=True)

# Se grafican los datos coloreados según el cultivo al que pertenecen
fig = px.scatter_3d(crop_recommendation_df, x='N', y='P', z='K',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'})
fig.update_layout(title = "Distribución original de los cultivos para las variables K, P y N")
fig.show(block=True)

"""Se obtendrá de manera más ordenada la lista de cultivos que pertenecen a cada cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""
for i in range(0,7):
    print('Cultivos en cluster', i, crop_recommendation_kmeans_df[crop_recommendation_kmeans_df['Cluster KMeans']==str(i)]['label'].unique())
    print('\n')


"""Cantidades de observaciones por cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""
observaciones_por_cluster = crop_recommendation_kmeans_df['Cluster KMeans'].value_counts().sort_index()
observaciones_por_cluster

"""Clusters con mayor cantidad de observaciones (Cluster 0 y 1), sugieren que es un grupo relativamente grande y heterogéneo en comparación 
con los demás clusters. Puede representar una categoría de cultivos común o podría haber una mayor variabilidad 
en las características agronómicas de los cultivos en este grupo.

Clusters con menos observaciones sugieren que los cultivos en este grupo pueden tener características más específicas 
que los hacen menos comunes en comparación con otros grupos.

Se obtienen las medias de cada cluster por columna
"""

crop_recommendation_kmeans_df_grouped = crop_recommendation_kmeans_df.groupby('Cluster KMeans').mean()
crop_recommendation_kmeans_df_grouped

crop_recommendation_kmeans_df.groupby('Cluster KMeans').sum().plot(kind='bar', figsize=(10, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Suma de cada columna por cluster')
plt.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""


"""Se procede ahora a determinar el número óptimo de clusters por el método GAP.
"""

gs = OptimalK(n_jobs=1, n_iter=5)
numero_clusters = gs(standardized_data, n_refs=300, cluster_array=np.arange(2,15)) #n_refs number of sample reference datasets to create
print('Optimal clusters: ', numero_clusters)

# Gap Statistics data frame
gs.gap_df[['n_clusters', 'gap_value']]

# Graficamos los gap values con respecto al número de clusters
plt.figure(figsize=(10,6))
plt.plot(gs.gap_df.n_clusters, gs.gap_df.gap_value, linewidth=2)
plt.scatter(gs.gap_df[gs.gap_df.n_clusters == numero_clusters].n_clusters,
            gs.gap_df[gs.gap_df.n_clusters == numero_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.axvline(numero_clusters, linestyle="--")
plt.show(block=True)

kmeans = KMeans(n_clusters=13)
kmeans.fit(standardized_data) #Entrenamos el modelo

# El metodo labels_ nos da a que cluster corresponde cada observacion
crop_recommendation_kmeans_df['Cluster KMeans'] = kmeans.labels_
crop_recommendation_kmeans_df.head()

"""Se graficarán los clusters obtenidos y debajo otro gráfico 3D mostrando los cultivos correspondientes."""

crop_recommendation_kmeans_df['Cluster KMeans'] = crop_recommendation_kmeans_df['Cluster KMeans'].astype(str)

# Diccionario de asignación de colores
colores_personalizados_13 = ["#F72585",  "#b5179e",  "#7209b7", "#560bad", "#3a0ca3", "#3f37c9", "#4361ee", "#4cc9f0",
                           "#007f5f",  "#2b9348",  "#80b918", "#d4d700", "#eeef20"]


# Se grafican los clusters
fig = px.scatter_3d(crop_recommendation_kmeans_df[['K', 'N', 'P']], x='K', y='N', z='P',
              color=crop_recommendation_kmeans_df['Cluster KMeans'], color_discrete_sequence=colores_personalizados_13, labels={'color': 'Cluster KMeans'})
fig.update_layout(title = "13 Clusters de Kmeans para las variables K, P y N")
fig.show(block=True)

# Se grafican los datos coloreados según el cultivo al que pertenecen
fig = px.scatter_3d(crop_recommendation_df, x='N', y='P', z='K',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'})
fig.update_layout(title = "Distribución original de los cultivos para las variables K, P y N")
fig.show(block=True)

"""Se obtendrá de manera más ordenada la lista de cultivos que pertenecen a cada cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""
for i in range(0,13):
    print('Cultivos en cluster', i, crop_recommendation_kmeans_df[crop_recommendation_kmeans_df['Cluster KMeans']==str(i)]['label'].unique())
    print('\n')


"""Cantidad de observaciones por cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""
observaciones_por_cluster = crop_recommendation_kmeans_df['Cluster KMeans'].value_counts().sort_index()
observaciones_por_cluster

"""Se obtienen las medias de cada cluster por columna"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""
crop_recommendation_kmeans_df_grouped = crop_recommendation_kmeans_df.groupby('Cluster KMeans').mean()
crop_recommendation_kmeans_df_grouped

crop_recommendation_kmeans_df.groupby('Cluster KMeans').sum().plot(kind='bar', figsize=(10, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Suma de cada columna por cluster')
plt.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 6 - K-Means"""
"""************************************************************************************************"""




"""  ----------------------------------
     ----------------------------------
         7 - Clustereing jerárquico
     ----------------------------------
     ----------------------------------
"""
"""
Aplicar clustering jerárquico y determinar cuál número sería el que mejor represente los datos. 
Utilizar el score de Silhouette y calcular el número óptimo de cluster por medio de GAP.
"""

# Creamos el dendrograma para encontrar el número óptimo de clusters
linkage = sch.linkage(standardized_data, method='ward')
dendrogram = sch.dendrogram(linkage)

plt.title('Dendograma')
plt.xlabel('Categorías')
plt.ylabel('Distancias euclideanas')
plt.show(block=True)

"""Como se dificulta visualizar el gráfico en su totalidad ya que sobre el final la cantidad de 
clusters es elevada se procede a recortarlo para una mejor visualización.

Para ello se observan las distancias registradas en el eje y y se establece un umbral de corte. 
Se observa que las distancias más largas y dónde se producen los mayores cambios 
en la altura (distancia) es alrededor del 15 por lo que se considera realizar el corte horizontalmente allí.

"""

# Genera el linkage y el dendrograma
linkage = sch.linkage(standardized_data, method='ward')

# Crea una figura con el tamaño deseado
fig, ax = plt.subplots(figsize=(10, 5))

dendrogram = sch.dendrogram(linkage, truncate_mode='lastp', p=15)

plt.title('Dendograma')
plt.xlabel('Categorías')
plt.ylabel('Distancias euclideanas')
plt.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""


n_clusters = 7
clustering = AgglomerativeClustering(n_clusters=n_clusters)

cluster_assignments = clustering.fit_predict(standardized_data)

crop_recommendation_jerarquico_df = crop_recommendation_df
crop_recommendation_jerarquico_df['Cluster_jerarquico'] = cluster_assignments

crop_recommendation_jerarquico_df.head()

"""Se utiliza la mérica de Silhouette para conocer la bondad de la técnica de agrupación."""
print(f'Silhouette score (n=7): {silhouette_score(standardized_data, cluster_assignments)}')

"""Un Silhouette Score de 0.3169 es una indicación positiva de la calidad de los clusters obtenidos 
a partir del conjunto de datos con 7 clusters. Esta lejos de ser un 1, que sería lo idea, pero aun así 
dicho valor sugiere que los cultivos están medianamente bien agrupados y definidos. Esto indica una 
estructura de clustering relativamente sólida en los datos.

Se graficarán los clusters obtenidos y debajo otro gráfico 3D mostrando los cultivos correspondientes.
"""

crop_recommendation_jerarquico_df['Cluster_jerarquico'] = crop_recommendation_jerarquico_df['Cluster_jerarquico'].astype(str)

# Diccionario de asignación de colores
colores_personalizados_7 = ["#F72585",  "#b5179e",  "#7209b7", "#560bad", "#3a0ca3", "#3f37c9", "#4361ee"]

# Se grafican los clusters
fig = px.scatter_3d(crop_recommendation_jerarquico_df[['K', 'N', 'P']], x='K', y='N', z='P',
              color=crop_recommendation_jerarquico_df['Cluster_jerarquico'], color_discrete_sequence=colores_personalizados_7, labels={'color': 'Cluster Agglomerative'})
fig.update_layout(title = "7 Clusters clústering jerárquico para las variables K, P y N")
fig.show(block=True)

# Se grafican los datos coloreados según el cultivo al que pertenecen
fig = px.scatter_3d(crop_recommendation_df, x='N', y='P', z='K',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'})
fig.update_layout(title = "Distribución original de los cultivos para las variables K, P y N")
fig.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""


"""Se obtendrá de manera más ordenada la lista de cultivos que pertenecen a cada cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
for i in range(0,7):
    print('Cultivos en cluster', i, crop_recommendation_jerarquico_df[crop_recommendation_jerarquico_df['Cluster_jerarquico']==str(i)]['label'].unique())
    print('\n')


"""Cantidad de observaciones por cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
observaciones_por_cluster = crop_recommendation_jerarquico_df['Cluster_jerarquico'].value_counts().sort_index()
observaciones_por_cluster


"""Se obtienen las medias de cada cluster por columna"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
crop_recommendation_kmeans_df_grouped = crop_recommendation_jerarquico_df.groupby('Cluster_jerarquico').mean()
crop_recommendation_kmeans_df_grouped

crop_recommendation_jerarquico_df.groupby('Cluster_jerarquico').sum().plot(kind='bar', figsize=(10, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Suma de cada columna por cluster')
plt.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""


n_clusters = 13
clustering = AgglomerativeClustering(n_clusters=n_clusters)

cluster_assignments = clustering.fit_predict(standardized_data)

crop_recommendation_jerarquico_df = crop_recommendation_df
crop_recommendation_jerarquico_df['Cluster_jerarquico'] = cluster_assignments

crop_recommendation_jerarquico_df.head()

print(f'Silhouette score (n=13): {silhouette_score(standardized_data, cluster_assignments)}')

"""El Silhouette score para 13 clustes es ligeramente mayor que para 7 clusters. 
Ha tenido una mejora mínima y la interpretación de dicho score es la misma que para 7 clusters.

Se graficarán los clusters obtenidos y debajo otro gráfico 3D mostrando los cultivos correspondientes.
"""

crop_recommendation_jerarquico_df['Cluster_jerarquico'] = crop_recommendation_jerarquico_df['Cluster_jerarquico'].astype(str)

# Diccionario de asignación de colores
colores_personalizados_13 = ["#F72585",  "#b5179e",  "#7209b7", "#560bad", "#3a0ca3", "#3f37c9", "#4361ee", "#4cc9f0",
                           "#007f5f",  "#2b9348",  "#80b918", "#d4d700", "#eeef20"]


# Se grafican los clusters
fig = px.scatter_3d(crop_recommendation_jerarquico_df[['K', 'N', 'P']], x='K', y='N', z='P',
              color=crop_recommendation_jerarquico_df['Cluster_jerarquico'], color_discrete_sequence=colores_personalizados_13, labels={'color': 'Cluster Agglomerative'})
fig.update_layout(title = "13 Clusters clústering jerárquico para las variables K, P y N")
fig.show(block=True)

# Se grafican los datos coloreados según el cultivo al que pertenecen
fig = px.scatter_3d(crop_recommendation_df, x='N', y='P', z='K',
              color=crop_recommendation_df['label'],  color_discrete_sequence=colores_personalizados, labels={'color': 'Cultivo'})
fig.update_layout(title = "Distribución original de los cultivos para las variables K, P y N")
fig.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""


"""Se obtendrá de manera más ordenada la lista de cultivos que pertenecen a cada cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
for i in range(0,13):
    print('Cultivos en cluster', i, crop_recommendation_jerarquico_df[crop_recommendation_jerarquico_df['Cluster_jerarquico']==str(i)]['label'].unique())
    print('\n')


"""Cantidad de observaciones por cluster"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
observaciones_por_cluster = crop_recommendation_jerarquico_df['Cluster_jerarquico'].value_counts().sort_index()
observaciones_por_cluster


"""Se obtienen las medias de cada cluster por columna"""
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""
crop_recommendation_kmeans_df_grouped = crop_recommendation_jerarquico_df.groupby('Cluster_jerarquico').mean()
crop_recommendation_kmeans_df_grouped

crop_recommendation_jerarquico_df.groupby('Cluster_jerarquico').sum().plot(kind='bar', figsize=(10, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Sume de cada columna por cluster')
plt.show(block=True)
"""************************************************************************************************"""
"""Observaciones pertinentes redactadas en el informe en la sección 7 - Clustereing jerárquico"""
"""************************************************************************************************"""

"""****************************************"""
"""Conclusiones redactadas en el informe"""
"""****************************************"""