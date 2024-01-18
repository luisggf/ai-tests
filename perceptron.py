# Implementando o Perceptron (Frank Rosenblatt, 1957)
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


# Definindo uma classe para o Perceptron
class Perceptron:

    def __init__(self, taxa_aprendizado=0.01, num_iteracoes=1000, tolerancia=10):
        self.alfa = taxa_aprendizado
        self.num_iteracoes = num_iteracoes
        self.func_ativacao = self.func_degrau
        self.pesos = None
        self.bias = None
        self.erros = []
        self.tolerancia = tolerancia

    def treina(self, X, y):
        n_amostras, n_caracteristicas = X.shape
        self.pesos = np.zeros(n_caracteristicas)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y]).astype(np.float64)

        erros_iteracao = []  # Armazenar os erros de cada iteração

        # O erro deve ser calculado para cada iteração individualmente.
        for _ in range(self.num_iteracoes):
            erro_iteracao = 0
            for idx, x_i in enumerate(X):

                saida_linear = np.dot(x_i, self.pesos) + self.bias
                y_previsto = self.func_ativacao(saida_linear)
                atualizacao = self.alfa * (y_[idx] - y_previsto)
                self.pesos += atualizacao * x_i
                self.bias += atualizacao
                erro_iteracao += int(atualizacao != 0.0)
            self.erros.append(erro_iteracao)
            erros_iteracao.append(erro_iteracao)
            print(f"Iteração: {_ + 1}, Erro: {erro_iteracao}")

            # Condição de parada: Encerra se o erro_iteracao se repetir nx consecutivas
            if erros_iteracao[-self.tolerancia:] == [erro_iteracao] * self.tolerancia:
                break

    def predicao(self, X):
        saida_linear = np.dot(X, self.pesos) + self.bias
        y_previsto = self.func_ativacao(saida_linear)
        return y_previsto

    def predicao_plot(self, X):
        pesos = []
        w2 = self.pesos[0]
        w1 = self.pesos[4]
        pesos.append(w1)
        pesos.append(w2)
        saida_linear = np.dot(X, pesos) + self.bias
        y_previsto = self.func_ativacao(saida_linear)
        return y_previsto

    def func_degrau(self, x):
        return np.where(x >= 0, 1, 0)

    def plota_erros(self):
        plt.plot(self.erros, marker='o', label='Erros por iteração')
        plt.xlabel('Iterações')
        plt.ylabel('Número de erros')
        plt.title('Erros de treinamento por iteração')
        plt.legend()
        plt.show()

    def plota_decisao(self, X, y):
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 4], c=y,
                   marker='o', label='Samples')
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 4]), np.max(X[:, 4])

        xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                             np.linspace(y_min, y_max))
        Z = self.predicao_plot(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='Paired')
        plt.xlabel("Age (Normalized)")
        plt.ylabel("Escolarity (Normalized)")
        plt.title("Decision Boundry")
        plt.show()


# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets


integer_columns = [0, 2, 4, 10, 11, 12]
nominal_columns = [col for col in X.columns if col not in integer_columns]

# preenche valores vazios de x com o valor mais comum da base naquela coluna
X.fillna(X.mode().iloc[0], inplace=True)


# normalização das colunas nominais (transformando-as em valores numericos)
label_encoder = LabelEncoder()
for col in nominal_columns:
    X[col] = label_encoder.fit_transform(X[col])


X = X.values

# normalizando conjunto de dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = y['income'].isin(['>50K', '<=50K']).astype(int)

# obtendo dados para calculo de metricas de desempenho
# valores reais esperados
y_true = y

# Inicializando e treinando o perceptron
perceptron = Perceptron(taxa_aprendizado=0.05,
                        num_iteracoes=100, tolerancia=10)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# treina rede neural antes de calcular valores previstos
perceptron.treina(X_train, y_train)

y_esperado = perceptron.predicao(X)

conf_matrix = confusion_matrix(y_true, y_esperado)

VP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
VN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

accuracy = (VP + VN) / len(X)
false_positive = FN / (FN + VN)
false_negative = FN / (FN + VP)

print('Precisão do modelo: ', accuracy)
print('Falsos positivos: ', false_positive)
print('Falsos negativos: ', false_negative)

# Gráfico do Limite de Decisão
perceptron.plota_decisao(X_test, y_test)

# Gráfico de erro por iteração
perceptron.plota_erros()


# Criar e visualizar dados simples 2D

# X, y = make_classification(n_samples=50, n_features=2, n_informative=2,
#                            n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=1)
# y = np.where(y == 0, -1, 1)  # Transforma os rótulos em 1 e -1


# # Inicializando e treinando o perceptron
# perceptron = Perceptron(taxa_aprendizado=0.01, num_iteracoes=20, tolerancia=0)
# perceptron.treina(X, y)


# # Gráfico do Limite de Decisão
# perceptron.plota_decisao(X, y)

# # Gráfico de erro por iteração
# perceptron.plota_erros()


# Atividade Prática -----------------------------------------------------------

# 1. Utilize um conjunto de dados linearmente separável, pesquise no repositório UCI Machine Learning.
# 2. Instancie a classe Perceptron criada e treine o modelo.
# 3. Utilize o método `plota_decisao` para visualizar a superfície de decisão que o modelo aprendeu.
# 4. Discuta em sala o que cada parâmetro (pesos e bias) do modelo significa e como eles influenciam na aprendizagem.
# 5. Experimente com diferentes taxas de aprendizado e número de iterações e observe as alterações na superfície de decisão.
