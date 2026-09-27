import numpy as np

"""
Neurónio artificial (RNA Parte 1 slides 12-22, Parte 2 32-33).
4 elementos: entradas x, pesos w com valores que representam a permeabilidade sináptica,
 função de transferência f e a saída y.
h = w·x + b (soma de ativação: produto interno + pendor, determina a deslocação da fronteira de discriminação
 , slide 17)
y = φ(h) (função de ativação: introduz não-linearidade, determina a saida do neurónio, 
resultante do produto escalar das entradas e dos pesos sinápticos somado com o pendor, slide 20-21)
w e b definem a fronteira de discriminação (recta/plano/hiperplano).


Adaptação (treino, Parte 3): o algoritmo de retropropagação
(Rumelhart, Hinton & Williams, 1986) minimiza a função de perda por
descida de gradiente, θ ← θ - α∇L(θ) (slide 8). Para cada neurónio j
de uma camada n, com componente de erro δ_j^n vindo da retropropagação
(slide 21) e a derivada y' = φ'(h) da sua função de ativação (slide 38),
a variação dos pesos e do pendor é dada por (slide 39):
    Δw = -α·y'·δ · y^(n-1)
    Δb = -α·y'·δ
    
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
            phi_derivada: derivada da função de ativação, φ': R → R
                          (usada no cálculo de y' para a retropropagação)
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
        self.y_derivada = 0  # y' = φ'(h), usada na retropropagação (slide 38)

    def propagar(self, x):
        """
        self.y_derivada = 0  # y' = φ'(h), usada na retropropagação (slide 38)

        Args:
            x: vetor das entradas
        """

        assert len(x) == self.d, f"Dimensão de entrada inválida: esperado {self.d}, recebido {len(x)}"

        # Cálculo da soma da ativação (h): pendor somado ao produto
        # interno entre os pesos e as entradas
        self.h = produto_interno(self.w, x) + self.b
        self.y = self.phi(self.h)  # aplicação da função de ativação (φ) à soma da ativação
        #eps = 1e-5
        #self.y_derivada = (self.phi(self.h + eps) - self.phi(self.h - eps)) / (2 * eps)
        self.y_derivada = 1 - self.y ** 2  # φ'(h) = 1 - tanh(h)², assume phi = tanh

        return self.y

    def adaptar(self, delta, yn_1, alpha, beta):
        """
            Adapta os pesos das ligações de entrada e o pendor interno do
            neurónio, com base na retropropagação do erro (slide 39, Parte 3).
            Args:
                delta: componente de propagação do erro de saída do neurónio (δ)
                yn_1: vetor de saída da camada anterior (y^(n-1))
                alpha: taxa de aprendizagem (α)
            """
        fator = -alpha * self.y_derivada * delta

        for i in range(len(self.w)):
            momento_w = beta * self.delta_w[i] # M_w = β·Δw_anterior
            self.delta_w[i] = fator * yn_1[i] + momento_w  # Δw_i = -α·y'·δ·y_i^(n-1) + M_w
            self.w[i] = self.w[i] + self.delta_w[i]  # w_i = w_i + Δw_i

        momento_b = beta * self.delta_b  # M_b = β·Δb_anterior
        self.delta_b = fator + momento_b  # Δb = -α·y'·δ + M_b
        self.b = self.b + self.delta_b  # b = b + Δb


