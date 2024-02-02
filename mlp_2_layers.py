from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLP_2LAYERS:
    def __init__(self, dim_entrada, dim_ocultos1, dim_ocultos2, dim_saida, taxa_aprendizado=0.01):
        self.dim_entrada = dim_entrada
        self.dim_ocultos1 = dim_ocultos1
        self.dim_ocultos2 = dim_ocultos2
        self.dim_saida = dim_saida
        self.taxa_aprendizado = taxa_aprendizado

        # Inicialização dos pesos e bias das camadas ocultos1, ocultos2, e de saída
        self.pesos_ocultos1 = np.random.rand(
            self.dim_entrada, self.dim_ocultos1)
        self.bias_ocultos1 = np.zeros(self.dim_ocultos1)

        self.pesos_ocultos2 = np.random.rand(
            self.dim_ocultos1, self.dim_ocultos2)
        self.bias_ocultos2 = np.zeros(self.dim_ocultos2)

        self.pesos_saida = np.random.rand(self.dim_ocultos2, self.dim_saida)
        self.bias_saida = np.zeros(self.dim_saida)

    # Funcao de Ativacao Sigmoide
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada Primeirada Funcao de Ativacao Sigmoide
    def sigmoide_derivada(self, x):
        return x * (1 - x)

    def feedforward(self, df):
        # Calculo da saida da 1ª Camada oculta
        camada_ocultos1_entrada = np.dot(
            df, self.pesos_ocultos1) + self.bias_ocultos1
        camada_ocultos1_saida = self.sigmoide(camada_ocultos1_entrada)

        # Calculo da saida da 2ª Camada oculta
        camada_ocultos2_entrada = np.dot(
            camada_ocultos1_saida, self.pesos_ocultos2) + self.bias_ocultos2
        camada_ocultos2_saida = self.sigmoide(camada_ocultos2_entrada)

        # Calculo da saida da Camada de saída
        camada_saida_entrada = np.dot(
            camada_ocultos2_saida, self.pesos_saida) + self.bias_saida
        saida = self.sigmoide(camada_saida_entrada)

        return camada_ocultos1_saida, camada_ocultos2_saida, saida

    def backpropagation(self, df, y, camada_ocultos1_saida, camada_ocultos2_saida, saida):
        # Backpropagation
        saida_erro = y - saida
        saida_delta = saida_erro * self.sigmoide_derivada(saida)

        camada_ocultos2_erro = saida_delta.dot(self.pesos_saida.T)
        camada_ocultos2_delta = camada_ocultos2_erro * \
            self.sigmoide_derivada(camada_ocultos2_saida)

        camada_ocultos1_erro = camada_ocultos2_delta.dot(self.pesos_ocultos2.T)
        camada_ocultos1_delta = camada_ocultos1_erro * \
            self.sigmoide_derivada(camada_ocultos1_saida)

        # Atualização dos pesos e bias
        self.pesos_saida += camada_ocultos2_saida.T.dot(
            saida_delta) * self.taxa_aprendizado
        self.bias_saida += np.sum(saida_delta, axis=0) * self.taxa_aprendizado

        self.pesos_ocultos2 += camada_ocultos1_saida.T.dot(
            camada_ocultos2_delta) * self.taxa_aprendizado
        self.bias_ocultos2 += np.sum(camada_ocultos2_delta,
                                     axis=0) * self.taxa_aprendizado

        self.pesos_ocultos1 += df.T.dot(camada_ocultos1_delta) * \
            self.taxa_aprendizado
        self.bias_ocultos1 += np.sum(camada_ocultos1_delta,
                                     axis=0) * self.taxa_aprendizado

    def fit(self, df, y, epocas=1000):
        for epoca in range(epocas):
            # Forward propagation
            camada_ocultos1_saida, camada_ocultos2_saida, saida = self.feedforward(
                df)
            # Backpropagation
            self.backpropagation(df, y, camada_ocultos1_saida,
                                 camada_ocultos2_saida, saida)
            erro_medio = np.mean(np.square(y - saida))
            print("MSE na época", epoca, ":", erro_medio)

    def predicao(self, df):
        _, _, saida = self.feedforward(df)
        return saida


def plot_values_distribution(y_pred):
    """
    Plots the distribution of predicted values after taking logarithm.
    Parameters:
    - y_pred: Numpy array of predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=10, color='blue', alpha=0.7)
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
x_data = x_data.astype(float).values

# separar conjunto de treino e conjunto de teste (test size setado para 97% pois valores maiores quebram o MLP)
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.97, shuffle=True, random_state=42)

# inicialização do MLP de 2 camadas
mlp = MLP_2LAYERS(dim_entrada=103, dim_ocultos1=5, dim_ocultos2=3,
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

plot_values_distribution(y_predicted)
print("Valores Continuos de Y previsto: ", y_predicted)


# filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
y_predicted_binary = (y_predicted > 0.5).astype(int)


# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_predicted_binary == y_test)

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
