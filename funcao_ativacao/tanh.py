import math
from funcao_ativacao.func_ativacao import FuncAtivacao

"""
Tangente hiperbólica (RNA Parte 1 slides 24 e 27, Parte 3 slide 31).
φ(h) = tanh(h), saída em [-1, 1], por isso serve para codificação bipolar
Derivada: φ'(h) = 1 - tanh(h)².
"""


class Tanh(FuncAtivacao):

    def calcular(self, h):
        return (math.exp(h) - math.exp(-h)) / (math.exp(h) + math.exp(-h))

    def derivada(self, h):
        tanh = self.calcular(h)
        return 1 - tanh ** 2
