import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo
from adult_prefiltration import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K


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
        return 1 / (1 + np.exp(-np.array(x, dtype=float)))

    # Derivada Primeirada Funcao de Ativacao Sigmoide
    def sigmoide_derivada(self, x):
        return x * (1 - x)

    # Calculo da saída da rede MLP
    def feedforward(self, df):
        # Calculo da saida da Camada oculta
        camada_oculta_entrada = np.dot(
            df, self.pesos_ocultos) + self.bias_ocultos
        camada_oculta_saida = self.sigmoide(camada_oculta_entrada)

        # Calculo da saida da Camada de saída
        camada_saida_entrada = np.dot(
            camada_oculta_saida, self.pesos_saida) + self.bias_saida
        saida = self.sigmoide(camada_saida_entrada)

        return camada_oculta_saida, saida

    def backpropagation(self, df, y, camada_oculta_saida, saida):
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

        self.pesos_ocultos += df.T.dot(camada_oculta_delta) * \
            self.taxa_aprendizado
        self.bias_ocultos += np.sum(camada_oculta_delta,
                                    axis=0) * self.taxa_aprendizado

    def fit(self, df, y, epocas=1000):
        for epoca in range(epocas):
            # Forward propagation
            camada_oculta_saida, saida = self.feedforward(df)
            # Backpropagation
            self.backpropagation(df, y, camada_oculta_saida, saida)

    def predicao(self, df):
        _, saida = self.feedforward(df)
        return saida

    # Função para plotar a superfície de decisão
    def plot_decision_surface(self, df, y):
        h = .02  # Passo da grade
        x_min, x_max = df[:, 0].min() - 1, df[:, 0].max() + 1
        y_min, y_max = df[:, 4].min() - 1, df[:, 4].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Calculo da saida da Camada oculta
        camada_oculta_entrada = np.dot(
            grid_points, self.pesos_ocultos) + self.bias_ocultos
        camada_oculta_saida = self.sigmoide(camada_oculta_entrada)
        Z = camada_oculta_saida.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plotar os pontos de treinamento
        plt.scatter(df[:, 0], df[:, 4], c=y,
                    cmap=cmap_bold, edgecolor='k', s=30)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Superfície de Decisão da Rede MLP")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
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


def normalize(columns):
    scaler = preprocessing.StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# obter dados por api
# adult = fetch_ucirepo(id=2)

# # data (as pandas dataframes)
# X = adult.data.features
# y = adult.data.targets


header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
          'capital-loss', 'hours-per-week', 'native-country', 'salary']
try:
    df_temp = pd.read_csv("adults.csv", index_col=False,
                          skipinitialspace=True, header=None, names=header)
except:
    df_temp = pd.read_csv("https://raw.githubusercontent.com/aliakbarbadri/mlp-classifier-adult-dataset/master/adults.csv",
                          index_col=False, skipinitialspace=True, header=None, names=header)

df = df_temp

# prefiltrations

# remove nans values
df = df.replace('?', np.nan)
print("Quantidade de valores NULL: ", df[pd.isnull(df).any(axis=1)].shape)


print("Tamanho, N° Colunas do Dataframe: ", df.shape)

# remover colunas contendo NAN
df.dropna(inplace=True)
print("Tamnho, N° de Colunas do Dataframe após remoção de dados nulos", df.shape)

# remover education-num porque existe coluna education nominal
df.drop('education-num', axis=1, inplace=True)

print(df.shape)

# colunas categoricas a serem normnalizadas
categorical_columns = ['workclass', 'education', 'marital-status',
                       'occupation', 'relationship', 'race', 'sex', 'native-country']
label_column = ['salary']

print(df['salary'])
convert_to_int(label_column)

print(df['salary'])
df = convert_to_onehot(df, categorical_columns)

normalize_columns = ['age', 'fnlwgt',
                     'capital-gain', 'capital-loss', 'hours-per-week']

# converter coluna salario em valores binarios

# valores antes e pos normalização
show_values(normalize_columns)

normalize(normalize_columns)

# pos normalização
show_values(normalize_columns)

y_labels = df['salary']
x_data = df.drop('salary', axis=1)

x_data = x_data.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.5, shuffle=True)


print(X_train.shape, y_train.shape)
print(y_test)
print(X_test.shape, y_test.shape)

# treina rede neural antes de calcular valores previstos
mlp = MLP(dim_entrada=103, dim_oculta=10, dim_saida=1, taxa_aprendizado=0.05)

y_train, y_test = y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

# treinamento da porção da base de entrada
mlp.fit(X_train, y_train, epocas=20)

# Realizando previsões no conjunto de teste
# y_esperado deve ser um array de arrays de valores entre 0 e 1
y_esperado = mlp.predicao(X_test)


# filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
y_esperado_binary = (y_esperado > 0.5).astype(int)


print(y_esperado_binary.shape)
print(y_test.shape)

# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_esperado_binary == y_test)
# f1_score = f1(y_test, y_esperado_binary)
# Assuming y_test and y_esperado_binary are ndarrays
y_test = y_test.flatten()  # Ensure y_test is a 1D array
# Ensure y_esperado_binary is a 1D array
y_esperado_binary = y_esperado_binary.flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_esperado_binary, labels=[0, 1])

# True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = cm.ravel()

# Accuracy
accuracy = accuracy_score(y_test, y_esperado_binary)

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Accuracy:", accuracy)
# OLD MAIN


# integer_columns = [0, 2, 4, 10, 11, 12]
# nominal_columns = [col for col in df.columns if col not in integer_columns]

# # preenche valores vazios de x com o valor mais comum da base naquela coluna
# df.fillna(df.mode().iloc[0], inplace=True)


# # normalização das colunas nominais (transformando-as em valores numericos)
# label_encoder = LabelEncoder()
# for col in nominal_columns:
#     df[col] = label_encoder.fit_transform(df[col])

# # df é um array de N valores numericos
# df = df.values

# # normalizando conjunto de dados
# scaler = StandardScaler()
# df = scaler.fit_transform(df)

# y = y['income'].isin(['>50K', '<=50K']).astype(int)

# # obtendo dados para calculo de metricas de desempenho
# # valores reais esperados

# # conveerte tabela de y em vetor de vetores com os valores de y
# y = y.values.reshape(-1, 1)
# y_true = y

# # Inicializando e treinando o mlp
# X_train, X_test, y_train, y_test = train_test_split(
#     df, y)

# # y_train, y_test = y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

# # treina rede neural antes de calcular valores previstos
# mlp = MLP(dim_entrada=14, dim_oculta=10, dim_saida=1, taxa_aprendizado=0.1)

# # treinamento da porção da base de entrada
# mlp.fit(X_train, y_train, epocas=1000)

# # Realizando previsões no conjunto de teste
# # y_esperado deve ser um array de arrays de valores entre 0 e 1
# y_esperado = mlp.predicao(df)

# # filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
# y_esperado_binary = (y_esperado > 0.5).astype(int)

# # conta quantidade de valores esperados iguais os valores de base
# accuracy = np.mean(y_esperado_binary == y_true)

# # conf_matrix = confusion_matrix(y_true, y_esperado_binary)


# print(X_test.shape, y_test.shape)

mlp.plot_decision_surface(X_test, y_test)
