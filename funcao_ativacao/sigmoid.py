from funcao_ativacao.func_ativacao import FuncAtivacao
import math

"""
Função sigmoide (RNA Parte 1 slide 24, Parte 3 slides 30-31).
φ(h) = 1 / (1 + e^-h), saída em [0, 1].
Derivada, que só depende da própria função:
φ'(h) = φ(h)(1 - φ(h)) (slide 30).
"""


class Sigmoid(FuncAtivacao):

    def calcular(self, h):
        return 1 / (1 + math.exp(-h))

    def derivada(self, h):
        phi = self.calcular(h)
        return phi * (1 - phi)
