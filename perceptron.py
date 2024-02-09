import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo
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
import seaborn as sns
from sklearn.metrics import accuracy_score

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

        # O erro deve ser calculado para cada iter  ação individualmente.
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


def plot_categorical_columns_distribution(data, categorical_columns):
    for colname in categorical_columns:
        plt.title('Column: ' + colname)

        (data[colname]
            .value_counts()
            .head(20)
            .plot(kind='barh'))

        plt.savefig(f"./graphics/{colname}_distribution.pdf")
        plt.show()


def correlation_between_numerical(data):
    # Select only numeric columns
    numeric_data = data.select_dtypes(include='number')

    # Calculate correlation
    corr = numeric_data.corr()

    # Plot heatmap
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values
    )

    plt.show()


def plot_numerical_columns_distribution(data):
    for colname, column_series in data.items:
        plt.title('Column: ' + colname)
        column_series.plot(kind='hist')
        plt.show()


def plot_log_y_values_distribution(y_pred):
    """
    Plots the distribution of predicted values after taking logarithm.

    Parameters:
    - y_pred: Numpy array of predicted values.
    """
    plt.title('Y Distribution')
    plt.hist(y_pred)
    plt.show()


def normalize(columns):
    scaler = preprocessing.StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


df = pd.read_csv('adult_filtered.csv')

# separar df em 2 grupos bem definidos
y_labels = df['salary']
x_data = df.drop('salary', axis=1)
x_data = x_data.astype(float).values


# separar conjunto de treino e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.30, shuffle=True, random_state=42)

# separar dataframe em 2 classes bem definidas
y_labels = df['salary']
x_data = df.drop('salary', axis=1)
x_data = x_data.astype(float).values


# inicializar modelo perceptron e seus parametros
perceptron = Perceptron(taxa_aprendizado=0.05,
                        num_iteracoes=100, tolerancia=50)


# treina rede neural antes de calcular valores previstos
perceptron.treina(X_train, y_train)

# realiza previsoes com conjunto de teste
y_esperado = perceptron.predicao(X_test)

# converter valores continuos em discretos
y_esperado_binary = (y_esperado > 0.5).astype(int)


# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_esperado_binary == y_test)

# cria matriz de confusão
cm = confusion_matrix(y_test, y_esperado_binary, labels=[0, 1])

# verdadeiro negativo, falso positivo, falso negativo, verdadeiro positivo
vn, fp, fn, vp = cm.ravel()

# calcula precisão baseado no y verdadeiro e o esperado
accuracy = accuracy_score(y_test, y_esperado_binary)

print("Verdadeiros Positivos:", vp)
print("Verdadeiros Negativos:", vn)
print("Falso Positivos:", fp)
print("Falso Negativos:", fn)
print("Precisão:", accuracy)

# # Gráfico do Limite de Decisão
perceptron.plota_decisao(X_test, y_esperado_binary)

# Gráfico de erro por iteração
perceptron.plota_erros()
