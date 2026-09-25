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
        self.h = 0  # soma de ativação das entradas, escalar com valor inicial de 0
        self.y = 0  # saída do neurónio

    def propagar(self, x):
        """
        Propaga o vetor de entradas x, gerando a saída do neurónio.

        Args:
            x: vetor das entradas
        """

        assert len(x) == self.d, f"Dimensão de entrada inválida: esperado {self.d}, recebido {len(x)}"

        # Cálculo da soma da ativação (h): pendor somado ao produto
        # interno entre os pesos e as entradas
        self.h = produto_interno(self.w, x) + self.b
        self.y = self.phi(self.h)  # aplicação da função de ativação (φ) à soma da ativação
        return self.y
