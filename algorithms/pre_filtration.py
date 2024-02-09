
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split


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


def show_values(df, columns):
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
    df = pd.read_csv("adult_filtered.csv")
except Exception as e:
    print('Não foi possível carregar arquivo de dados. Erro: ', e)

show_values(df, categorical_columns)
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


def normalize(columns):
    scaler = preprocessing.StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


# valores antes e pos normalização
show_values(normalize_columns)
normalize(normalize_columns)
# pos normalização
show_values(normalize_columns)

# separar dataframe em 2 classes bem definidas
y_labels = df['salary']
x_data = df.drop('salary', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.97, shuffle=True, random_state=42)

# treina rede neural antes de calcular valores previstos
mlp = MLP(dim_entrada=103, dim_oculta=10, dim_saida=1, taxa_aprendizado=0.1)


y_train, y_test = y_train.values.reshape(
    -1, 1), y_test.values.reshape(-1, 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# treinamento da porção da base de entrada
mlp.fit(X_train, y_train, epocas=100)

# Realizando previsões no conjunto de teste
# y_esperado deve ser um array de arrays de valores entre 0 e 1
y_esperado = mlp.predicao(X_test)

# plot_log_y_values_distribution(y_esperado)

print(y_esperado)
# print(y_esperado)

# filtra probabilidades entre 0 e 1 sendo 1 > 0.5 e 0 <= 0.5
y_esperado_binary = (y_esperado > 0.5).astype(int)


print(y_esperado_binary.shape)
print(y_test.shape)

# conta quantidade de valores esperados iguais os valores de base
accuracy = np.mean(y_esperado_binary == y_test)


y_test = y_test.flatten()
y_esperado_binary = y_esperado_binary.flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_esperado_binary, labels=[0, 1])

# True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = cm.ravel()

print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
