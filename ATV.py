import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pickle import dump, load
from sklearn.cluster import KMeans

#Carregando e balanceando a base
df = pd.read_csv('iris.csv', sep=';')
print(df.head())
print(df['class'].value_counts())

#Normalizando a base
#Separar dados numéricos e categóricos
dados_num = df.drop(columns=['class'])
dados_cat = df['class']
dados_cat_rotulos = 'class'

#MinMax para os numéricos
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)

#Salvar o modelo normalizador
dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

#Transformar os dados numéricos
dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

#Get dummies para os categóricos
dados_cat_normalizado = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)

#Juntar tudo em um DataFrame único com o .join()
df_normalizado = dados_num_normalizados.join(dados_cat_normalizado)
print(df_normalizado.head())

#Determinando o número ótimo de clusters (Método do Cotovelo)
inercia = []
K = range(1, 11)

for k in K:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans.fit(df_normalizado)
  inercia.append(kmeans.inertia_)

#Plotando o gráfico do método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(K, inercia, 'bx-', marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Definir o k Ótimo')
plt.grid(True)
plt.show()

#Treinando o K-Means e salvando com o dump
k_otimo = 3
modelo_kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
modelo_kmeans.fit(df_normalizado)

dump(modelo_kmeans, open('modelo_kmeans_iris.model', 'wb'))

#Nova instância, predição e desnormalização
#Escolher uma linha aleatória da base de dados usando .sample()
instancia_aleatoria = dados_num.sample(n=1)
print('\nValores originais da instância sorteada:')
print(instancia_aleatoria)

#Normalizar a nova instância usando o transform() do modelo salvo
instancia_normalizada = modelo_normalizador.transform(instancia_aleatoria)

#Alinhando com as colunas completas do dataframe normalizado
nova_flor_completa = pd.DataFrame(0, index=[0], columns=df_normalizado.columns)
for i, col in enumerate(dados_num.columns):
  nova_flor_completa[col] = instancia_normalizada[0][i]

#Carregar o modelo KMeans usando o load() e prever o cluster
modelo_carregado = load(open('modelo_kmeans_iris.model', 'rb'))
grupo_previsto = modelo_carregado.predict(nova_flor_completa)
print(f'Cluster previsto: {grupo_previsto[0]}')

#Desnormalizar o centróide daquele cluster correspondente
centroide_normalizado = modelo_carregado.cluster_centers_[grupo_previsto[0]]
centroide_num_scaled = centroide_normalizado[:4].reshape(1, -1)
centroide_original = modelo_normalizador.inverse_transform(centroide_num_scaled)

print('Valores originais do centróide do grupo:')
print(centroide_original)