import numpy as np


def produto_interno(a, b):
    """Produto interno, para futuras medidas de semelhança."""
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

        # Vetor de pesos de tamanho d. Uso de uniform pois pesos precisam de ser contínuos para ajuste de pequenos incrementos
        self.w = [np.random.uniform(-1, 1) for _ in range(self.d)]
        self.b = np.random.uniform(-1, 1)  # pendor interno dos neurónios
        self.h = 0  # soma de ativação das entradas, escalar com valor inicial de 0
        self.y = 0  # saída do neurónio

    def propagar(self, x):
        """
        Propaga o vetor de entradas x, gerando a saída do neurónio.

        Args:
            x: vetor das entradas
        """
        # Cálculo da soma da ativação (h): pendor somado ao produto
        # interno entre os pesos e as entradas
        self.h = produto_interno(self.w, x) + self.b
        self.y = self.phi(self.h)  # aplicação da função de ativação (φ) à soma da ativação
        return self.y
