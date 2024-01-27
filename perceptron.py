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


def normalize(columns):
    scaler = preprocessing.StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
          'capital-loss', 'hours-per-week', 'native-country', 'salary']
try:
    df_temp = pd.read_csv("adults.csv", index_col=False,
                          skipinitialspace=True, header=None, names=header)
except:
    df_temp = pd.read_csv("https://raw.githubusercontent.com/aliakbarbadri/mlp-classifier-adult-dataset/master/adults.csv",
                          index_col=False, skipinitialspace=True, header=None, names=header)

df = df_temp

# prefiltragens

# remover valores nans
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

perceptron = Perceptron(taxa_aprendizado=0.01,
                        num_iteracoes=100, tolerancia=50)


# treina rede neural antes de calcular valores previstos
perceptron.treina(X_train, y_train)

y_esperado = perceptron.predicao(X_test)

y_esperado_binary = (y_esperado > 0.5).astype(int)


print(y_esperado_binary.shape)
print(y_test.shape)

# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_esperado_binary == y_test)

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


# # Gráfico do Limite de Decisão
perceptron.plota_decisao(X_test, y_esperado_binary)

# Gráfico de erro por iteração
perceptron.plota_erros()


# Atividade Prática -----------------------------------------------------------

# 1. Utilize um conjunto de dados linearmente separável, pesquise no repositório UCI Machine Learning.
# 2. Instancie a classe Perceptron criada e treine o modelo.
# 3. Utilize o método `plota_decisao` para visualizar a superfície de decisão que o modelo aprendeu.
# 4. Discuta em sala o que cada parâmetro (pesos e bias) do modelo significa e como eles influenciam na aprendizagem.
# 5. Experimente com diferentes taxas de aprendizado e número de iterações e observe as alterações na superfície de decisão.
