
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

        # Calculo da saída da rede MLP

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
            print("MSE na época", epoca, ":", erro_medio)

    def predicao(self, X):
        _, saida = self.feedforward(X)
        return saida

    # Função para plotar a superfície de decisão

    def plot_decision_surface(self, X, y):
        h = .02  # Passo da grade
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = self.predicao(grid_points)
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plotar os pontos de treinamento
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)
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


def plot_values_distribution(y_pred):
    """
    Plots the distribution of predicted values after taking logarithm.

    Parameters:
    - y_pred: Numpy array of predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=30, color='blue', alpha=0.7)
    plt.title('Distribuição dos valores previstos')
    plt.xlabel('Valores Continuos de Y')
    plt.ylabel('Frequência do Valor')
    plt.grid(True)
    plt.show()


# # ler dados já filtrados por praticidade
df = pd.read_csv('adult_filtered.csv')

# separar classes majoritarias e menoritarias
majority_class = 0
minority_class = 1

majority_data = df[df['salary'] == majority_class]
minority_data = df[df['salary'] == minority_class]

# downsample na classe majoritaria 0
downsampled_majority_data = resample(majority_data,
                                     replace=False,
                                     n_samples=len(minority_data),
                                     random_state=42)  # usamos seed 42 para ter resultados reproduziveis


# combinar classes reduzidas
downsampled_data = pd.concat([downsampled_majority_data, minority_data])

# embaralhar classe para nao atrapalhar treinamento
df = downsampled_data.sample(
    frac=1, random_state=42).reset_index(drop=True)

# separar df em 2 grupos bem definidos
y_labels = df['salary']
x_data = df.drop('salary', axis=1)

# separar conjunto de treino e conjunto de teste (test size setado para 97% pois valores maiores quebram o MLP)
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.97, shuffle=True, random_state=42)


# inicializando MLP de 1 camada
mlp = MLP(dim_entrada=103, dim_oculta=5,
          dim_saida=1, taxa_aprendizado=0.01)

# previamente y é um vetor de vetores, para treinar queremos 1 coluna de 0s e 1s
# então convertemos pelo .values (transformar em vetor) .reshape (moldar coluna)
y_train, y_test = y_train.values.reshape(
    -1, 1), y_test.values.reshape(-1, 1)


# mostra o formato dos parametros de teste e treino
print("Parâmetros de treino: ", X_train.shape, y_train.shape)
print("Parâmetros de teste: ", X_test.shape, y_test.shape)


# treinamento da porção da base de entrada
mlp.fit(X_train, y_train, epocas=1000)

# Realizando previsões no conjunto de teste
# y_predicted deve ser um array de arrays de valores entre 0 e 1
y_predicted = mlp.predicao(X_test)

# mostra distribuição de Y
plot_values_distribution(y_predicted)
print("Valores Continuos de Y previsto: ", y_predicted)


# filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
y_predicted_binary = (y_predicted > 0.5).astype(int)


# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_predicted_binary == y_test)

y_predicted_binary_bckp = y_predicted_binary

# achata o formato de y_predicted
y_predicted_binary = y_predicted_binary.flatten()

# inicializa matriz de confusão
cm = confusion_matrix(y_test, y_predicted_binary, labels=[0, 1])

# True Positives, True Negatives, False Positives, False Negatives
vn, fp, fn, vp = cm.ravel()

print("Precisão:", accuracy)
print("Verdadeiros Positivos:", vp)
print("Verdadeiros Negativos:", vn)
print("Falsos positivos:", fp)
print("Falsos negativos:", fn)
