import numpy as np

"""
Neurónio artificial (RNA Parte 1 slides 12-22, Parte 2 slides 32-33, Parte 3 slides 37-39
sem momento; com momento: Parte 4 slides 38-39).
Quatro elementos: entradas x, pesos w com valores que representam a permeabilidade sináptica,
função de transferência f e a saída y.
h = w*x + b (soma de ativação: produto interno + pendor, determina a deslocação da fronteira de discriminação, slide 17)
y = φ(h) (função de ativação: introduz não-linearidade, determina a saida do neurónio, 
resultante do produto escalar das entradas e dos pesos sinápticos somado com o pendor, slide 20-21)
w e b definem a fronteira de discriminação (recta/plano/hiperplano).
 
Para treinar, o neurónio guarda também y' = φ'(h) = dφ/dh (Parte 3 slides 37-38),
que a retropropagação usa tanto para adaptar os pesos como para passar o erro à
camada anterior. φ é um objeto FuncAtivacao (Sigmoid, Tanh) que sabe
calcular φ(h) e φ'(h), por isso o neurónio nunca precisa de saber qual é.
 
Adaptação (treino, Parte 3)- o algoritmo de retropropagação
(Rumelhart, Hinton & Williams, 1986) minimiza a função de perda por
descida de gradiente, θ ← θ - α∇L(θ) (slide 8). Para cada neurónio j
de uma camada n, com componente de erro δ_j^n vindo da retropropagação
(slide 21) e a derivada y' = φ'(h) da sua função de ativação (slide 38),
a variação dos pesos e do pendor é dada por (slide 39):
    Δw = -α*y'*δ * y^(n-1)
    Δb = -α*y'*δ
β é o factor de momento, com β=0 corresponde à descida de gradiente
simples, onde valores muito altos como β=2.0 podem
introduzir oscilações e instabilidade (slide 13).
"""


def produto_interno(a, b):
    """Produto interno (escalar) entre dois vetores, usado no cálculo da
    soma de ativação h.
    Expressa uma medida de semelhança entre o vetor de entrada e o vetor dos pesos."""

    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


class Neuronio:
    """Neurónio artificial: pesos, pendor e função de ativação."""

    def __init__(self, d, phi):
        """
        Args:
            d: dimensão de entrada. d ∈ Z+
            phi: função de ativação. φ: R → R
        """
        self.d = d
        self.phi = phi

        # Vetor de pesos de dimensão d, inicializado aleatoriamente no intervalo [-1, 1]
        self.w = [np.random.uniform(-1, 1) for _ in range(self.d)]
        self.b = np.random.uniform(-1, 1)  # pendor interno do neurónio

        self.delta_w = [0 for _ in range(self.d)]  # variação dos pesos (Δw), guardada para o termo de momento
        self.delta_b = 0  # variação do pendor (Δb), guardada para o termo de momento

        self.h = 0  # soma de ativação das entradas, escalar com valor inicial de 0
        self.y = 0  # saída do neurónio
        self.y_derivada = 0  # y' = φ'(h), calculada em propagar() e usada na retropropagação

    def propagar(self, x):
        """
        Calcula a saída do neurónio e a respetiva derivada (slide 38, Parte 3).

        Args:
            x: vetor das entradas
        Returns:
            A saída y do neurónio.
        """

        self.h = produto_interno(self.w, x) + self.b  # h = w*x + b
        self.y = self.phi.calcular(self.h)  # y = φ(h)
        self.y_derivada = self.phi.derivada(self.h)  # y' = φ'(h): é em h que se avalia, não em y nem em x

        return self.y

    def adaptar(self, delta, yn_1, alpha, beta):
        """
        Ajusta os pesos e o pendor com base no erro retropropagado (slide 39, Parte 3).
        Tem de ser chamado após propagar(), porque usa o y' guardado lá.

        Args:
            delta: componente de propagação do erro de saída do neurónio (δ)
            yn_1: vetor de saída da camada anterior (y^(n-1))
            alpha: taxa de aprendizagem (α)
            beta: momento

        """
        fator = -alpha * self.y_derivada * delta  # parte comum a Δw e Δb: -α*y'*δ

        for i in range(len(self.w)):
            momento_w = beta * self.delta_w[i]
            self.delta_w[i] = fator * yn_1[i] + momento_w
            self.w[i] = self.w[i] + self.delta_w[i]

        momento_b = beta * self.delta_b
        self.delta_b = fator + momento_b
        self.b = self.b + self.delta_b
