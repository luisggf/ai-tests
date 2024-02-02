
from sklearn.utils import resample
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
# from keras import backend as K
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# import keras
# from keras.metrics import Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, BinaryAccuracy, BinaryCrossentropy


class MLP:
    def __init__(self, dim_entrada, dim_oculta, dim_saida, taxa_aprendizado=0.01):
        self.dim_entrada = dim_entrada
        self.dim_oculta = dim_oculta
        self.dim_saida = dim_saida
        self.taxa_aprendizado = taxa_aprendizado

        # Inicialização dos pesos e bias das camadas oculta e de saída
        self.pesos_ocultos = np.random.rand(self.dim_entrada, self.dim_oculta)
        self.bias_ocultos = np.zeros(self.dim_oculta)

        self.pesos_saida = np.random.rand(self.dim_oculta, self.dim_saida)
        self.bias_saida = np.zeros(self.dim_saida)

    # Funcao de Ativacao Sigmoide
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada Primeirada Funcao de Ativacao Sigmoide
    def sigmoide_derivada(self, x):
        return x * (1 - x)

    # Calculo da saída da rede MLP
    def feedforward(self, X):
        # Calculo da saida da Camada oculta
        camada_oculta_entrada = np.dot(
            X, self.pesos_ocultos) + self.bias_ocultos
        camada_oculta_saida = self.sigmoide(camada_oculta_entrada)

        # Calculo da saida da Camada de saída
        camada_saida_entrada = np.dot(
            camada_oculta_saida, self.pesos_saida) + self.bias_saida
        saida = self.sigmoide(camada_saida_entrada)

        return camada_oculta_saida, saida

    def backpropagation(self, X, y, camada_oculta_saida, saida):
        # Backpropagation
        saida_erro = y - saida
        saida_delta = saida_erro * self.sigmoide_derivada(saida)

        camada_oculta_erro = saida_delta.dot(self.pesos_saida.T)
        camada_oculta_delta = camada_oculta_erro * \
            self.sigmoide_derivada(camada_oculta_saida)

        # Atualização dos pesos e bias
        self.pesos_saida += camada_oculta_saida.T.dot(
            saida_delta) * self.taxa_aprendizado
        self.bias_saida += np.sum(saida_delta, axis=0) * self.taxa_aprendizado

        self.pesos_ocultos += X.T.dot(camada_oculta_delta) * \
            self.taxa_aprendizado
        self.bias_ocultos += np.sum(camada_oculta_delta,
                                    axis=0) * self.taxa_aprendizado

    def fit(self, df, y, epocas=1000):
        for epoca in range(epocas):
            # Forward propagation
            camada_oculta1_saida, saida = self.feedforward(
                df)
            # Backpropagation
            self.backpropagation(df, y, camada_oculta1_saida, saida)
            erro_medio = np.mean(np.square(y - saida))
            print("Mean Squared Error for epoch", epoca, ":", erro_medio)

    def predicao(self, X):
        _, saida = self.feedforward(X)
        return saida
    # Função para plotar a superfície de decisão


def plot_decision_surface(self, df, y):
    h = .02  # Passo da grade
    x_min, x_max = df[:, 0].min() - 1, df[:, 0].max() + 1
    y_min, y_max = df[:, 1].min() - 1, df[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Calculo da saida da Camada oculta
    camada_oculta_entrada = np.dot(
        grid_points, self.pesos_oculto) + self.bias_oculto
    camada_oculta_saida = self.sigmoide(camada_oculta_entrada)
    Z = camada_oculta_saida.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plotar os pontos de treinamento
    plt.scatter(df[:, 0], df[:, 1], c=y,
                cmap=cmap_bold, edgecolor='k', s=30)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Superfície de Decisão da Rede MLP")
    plt.xlabel("Age (Normalized)")
    plt.ylabel("Capital Gain (Normalized)")
    plt.show()


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


def normalize(df, columns):
    scaler = preprocessing.MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])


def convert_bool_to_int(df, columns):
    for column in columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
    return df


def plot_log_y_values_distribution(y_pred):
    """
    Plots the distribution of predicted values after taking logarithm.

    Parameters:
    - y_pred: Numpy array of predicted values.
    """
    # Take the logarithm of predicted values, handling zero values
    log_y_pred = np.log1p(y_pred)

    plt.figure(figsize=(10, 6))
    plt.hist(log_y_pred, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Logarithm of Predicted Values')
    plt.xlabel('Logarithm of Predicted Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Function to plot decision surface


# def plot_decision_surface_keras(X, y):
#     h = .02  # Step size of the mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # Predict using the model
#     Z = model.predict(grid_points)
#     Z = (Z > 0.5).astype(int)

#     # Reshape the predictions for contour plot
#     Z = Z.reshape(xx.shape)

#     # Plot decision surface
#     plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.RdBu, edgecolors='k', s=30)
#     plt.xlabel('Age')
#     plt.ylabel('Capital Gain')
#     plt.title('Decision Surface of MLP Model')
#     plt.show()


# # ler dados já filtrados por praticidade
df = pd.read_csv('adults_filtered_(0-1).csv')


# Separate majority and minority classes
majority_class = 0
minority_class = 1

majority_data = df[df['salary'] == majority_class]
minority_data = df[df['salary'] == minority_class]

# Downsample majority class
downsampled_majority_data = resample(majority_data,
                                     replace=False,  # sample without replacement
                                     # match minority class
                                     n_samples=len(minority_data),
                                     random_state=42)  # reproducible results


# Combine minority class with downsampled majority class
downsampled_data = pd.concat([downsampled_majority_data, minority_data])

# Shuffle the dataset
df = downsampled_data.sample(
    frac=1, random_state=42).reset_index(drop=True)

# df = df[['age', 'capital-gain', 'hours-per-week', 'salary']]

# separar df em 2 grupos bem definidos
y_labels = df['salary']
x_data = df.drop('salary', axis=1)
x_data = x_data.astype(float).values

# separar conjunto de treino e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.2, shuffle=True, random_state=42)

# treina rede neural antes de calcular valores previstos
mlp = MLP(dim_entrada=103, dim_oculta=15,
          dim_saida=1, taxa_aprendizado=0.01)


y_train, y_test = y_train.values.reshape(
    -1, 1), y_test.values.reshape(-1, 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# treinamento da porção da base de entrada
mlp.fit(X_train, y_train, epocas=500)

# Realizando previsões no conjunto de teste
# y_esperado deve ser um array de arrays de valores entre 0 e 1
y_esperado = mlp.predicao(X_test)

# plot_log_y_values_distribution(y_esperado)

print(y_esperado)


# filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
y_esperado_binary = (y_esperado > 0.5).astype(int)


# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_esperado_binary == y_test)


y_esperado_binary = y_esperado_binary.flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_esperado_binary, labels=[0, 1])

# True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = cm.ravel()

# Accuracy
accuracy = accuracy_score(y_test, y_esperado_binary)

print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)


# modelo construido com Keras
# construir modelo MLP sequencial com lib Keras
# model = Sequential()
# model.add(Dense(100, input_dim=103, activation='relu'))
# # 1 camada oculta
# model.add(Dense(50, activation='relu'))
# # 2 camada oculta se necessário (caso for utilizar 1 para métricas #comentar)
# model.add(Dense(25, activation='relu'))
# # camada de saída com ativação sigmoid
# model.add(Dense(1, activation='sigmoid'))

# # compilar modelo com classificação binaria e diversas metricas de performance
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy', Precision(), Recall(), AUC(), TruePositives(
#     ), TrueNegatives(), FalsePositives(), FalseNegatives(), BinaryAccuracy()]
# )

# # treinar modelo
# model.fit(X_train, y_train, epochs=10, batch_size=32,
#           validation_data=(X_test, y_test))

# # avaliar modelo
# accuracy = model.evaluate(X_test, y_test)[1]
# print(f"Accuracy: {accuracy}")

# # obter valor predito de y continuo
# y_pred_probs = model.predict(X_test)

# # converter em valor discreto
# y_pred_binary = (y_pred_probs > 0.5).astype(int)

# # concatenar age e capital-gain em um array 2d

# plot_decision_surface_keras(X_test, y_pred_binary)
