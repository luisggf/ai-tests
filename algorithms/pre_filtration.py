
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import keras


def convert_to_int(columns):
    for column in columns:
        unique_values = df[column].unique().tolist()
        dic = {}
        for indx, val in enumerate(unique_values):
            dic[val] = indx
        df[column] = df[column].map(dic).astype(int)
        print(column + " done!")


def convert_to_onehot(data, columns):
    dummies = pd.get_dummies(data[columns])
    data = data.drop(columns, axis=1)
    data = pd.concat([data, dummies], axis=1)
    return data


def show_values(columns):
    for column in columns:
        max_val = df[column].max()
        min_val = df[column].min()
        mean_val = df[column].mean()
        var_val = df[column].var()
        print(column + ': values=['+str(min_val)+','+str(max_val) +
              '] , mean='+str(mean_val)+' , var='+str(var_val))


def normalize(columns):
    scaler = preprocessing.StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


def convert_bool_to_int(df, columns):
    for column in columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
    return df


header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
          'capital-loss', 'hours-per-week', 'native-country', 'salary']
# colunas categoricas a serem normnalizadas
categorical_columns = ['workclass', 'education', 'marital-status',
                       'occupation', 'relationship', 'race', 'sex', 'native-country']
label_column = ['salary']
normalize_columns = ['age', 'fnlwgt',
                     'capital-gain', 'capital-loss', 'hours-per-week']

try:
    df = pd.read_csv("adults.csv", index_col=False,
                     skipinitialspace=True, header=None, names=header)
except Exception as e:
    print('Não foi possível carregar arquivo de dados. Erro: ', e)


# prefiltragens

# remover valores nans
df = df.replace('?', np.nan)
print("Quantidade de valores NULL: ", df[pd.isnull(df).any(axis=1)].shape)

# formato do df
print("Tamanho, N° Colunas do Dataframe: ", df.shape)

# remover colunas contendo NAN
df.dropna(inplace=True)
print("Tamnho, N° de Colunas do Dataframe após remoção de dados nulos", df.shape)

# remover education-num porque existe coluna education nominal
df.drop('education-num', axis=1, inplace=True)

# como função convert_to_int alterou coluna salario
print("Coluna salário antes: ", df['salary'])
# converter coluna salario em valores binarios
convert_to_int(label_column)
print("Coluna salário depois: ", df['salary'])

# converter colunas categoricas em valores numericos
df = convert_to_onehot(df, categorical_columns)

# valores antes e pos normalização
show_values(normalize_columns)
normalize(normalize_columns)
# pos normalização
show_values(normalize_columns)

# separar dataframe em 2 classes bem definidas
y_labels = df['salary']
x_data = df.drop('salary', axis=1)


# # treina rede neural antes de calcular valores previstos
# mlp = MLP(dim_entrada=103, dim_oculta=10, dim_saida=1, taxa_aprendizado=0.1)


# y_train, y_test = y_train.values.reshape(
#     -1, 1), y_test.values.reshape(-1, 1)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# # treinamento da porção da base de entrada
# mlp.fit(X_train, y_train, epocas=100)

# # Realizando previsões no conjunto de teste
# # y_esperado deve ser um array de arrays de valores entre 0 e 1
# y_esperado = mlp.predicao(X_test)

# # plot_log_y_values_distribution(y_esperado)

# print(y_esperado)
# # print(y_esperado)

# # filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
# y_esperado_binary = (y_esperado > 0.5).astype(int)


# print(y_esperado_binary.shape)
# print(y_test.shape)

# # conta quantidade de valores esperados iguais os valores de base
# accuracy = np.mean(y_esperado_binary == y_test)


# y_test = y_test.flatten()
# y_esperado_binary = y_esperado_binary.flatten()

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_esperado_binary, labels=[0, 1])

# # True Positives, True Negatives, False Positives, False Negatives
# tn, fp, fn, tp = cm.ravel()

# # Accuracy
# accuracy = accuracy_score(y_test, y_esperado_binary)

# print("Accuracy:", accuracy)
# print("True Positives:", tp)
# print("True Negatives:", tn)
# print("False Positives:", fp)
# print("False Negatives:", fn)
