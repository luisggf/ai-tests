# Implementando o Perceptron (Frank Rosenblatt, 1957)
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


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
        # self.pesos = np.random.rand(n_caracteristicas)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y]).astype(np.float64)

        erros_iteracao = []  # Armazenar os erros de cada iteração
        repeticoes_erro = 0

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

    def func_degrau(self, x):
        return np.where(x >= 0, 1, 0)

    def plota_decisao(self, X, y):
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   marker='o', label='Amostras')
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                             np.linspace(y_min, y_max))
        Z = self.predicao(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='Paired')
        plt.xlabel("Caracteristica 1")
        plt.ylabel("Caracteristica 2")
        plt.title("Limite de Decisão")
        plt.show()

    def plota_erros(self):
        plt.plot(self.erros, marker='o', label='Erros por iteração')
        plt.xlabel('Iterações')
        plt.ylabel('Número de erros')
        plt.title('Erros de treinamento por iteração')
        plt.legend()
        plt.show()


# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# columns from df that are integer type (pre filtration to train the model)
# integer_columns = [0, 2, 4, 10, 11, 12]
integer_columns = [0, 4]

# Use the specified columns
X = X.iloc[:, integer_columns]
X = X.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = y['income'].isin(['>50K', '<=50K']).astype(int)


# Inicializando e treinando o perceptron
perceptron = Perceptron(taxa_aprendizado=0.01,
                        num_iteracoes=100, tolerancia=20)


X_train, X_test, y_train, y_test = train_test_split(X, y)


perceptron.treina(X_train, y_train)


# Gráfico do Limite de Decisão
perceptron.plota_decisao(X_test, y_test)

# Gráfico de erro por iteração
perceptron.plota_erros()


# Criar e visualizar dados simples 2D

# X, y = make_classification(n_samples=50, n_features=2, n_informative=2,
#                            n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=1)
# y = np.where(y == 0, -1, 1)  # Transforma os rótulos em 1 e -1

# print(X)
# print("AAAAAAAAAAAAAAAAAAAAAAAA\n", y)

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
