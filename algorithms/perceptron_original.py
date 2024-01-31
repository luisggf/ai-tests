import numpy as np


class Perceptron:
    def __init__(self, dim_entrada, taxa_aprendizado=0.01, epocas=100):
        self.dim_entrada = dim_entrada
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = np.random.rand(dim_entrada + 1)  # +1 para o bias
        self.erros = []

    # Funcao Degrau
    def funcao_ativacao(self, x):
        return 1 if x >= 0 else 0

    # Funcao de Predicao
    def predicao(self, X):
        X_with_bias = np.c_[X, np.ones(len(X))]
        return np.array([self.funcao_ativacao(np.dot(x, self.pesos)) for x in X_with_bias])

    # Funcao de treinamento
    def treina(self, X, y):
        X_with_bias = np.c_[X, np.ones(len(X))]
        for _ in range(self.epocas):
            total_erro = 0
            for i in range(len(X_with_bias)):
                erro = y[i] - \
                    self.funcao_ativacao(np.dot(X_with_bias[i], self.pesos))
                self.pesos += self.taxa_aprendizado * erro * X_with_bias[i]
                total_erro += int(erro != 0)
            self.erros.append(total_erro)


# Exemplo de uso (AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(dim_entrada=2, taxa_aprendizado=0.1, epocas=100)
perceptron.treina(X, y)

# Testando a previsão
print("Esperado:", y)
print("Previsão:", perceptron.predicao(X))

# Pesos
print("Pesos sinápticos:", perceptron.pesos)
