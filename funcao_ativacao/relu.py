from funcao_ativacao.func_ativacao import FuncAtivacao

"""
ReLU, Rectified Linear Unit (RNA Parte 1 slide 24): φ(h) = h se h > 0, senão 0.
Linear por partes, a derivada é 1 para h > 0 e 0 para h <= 0 (em h = 0 usa-se 0).
"""


class ReLU(FuncAtivacao):

    def calcular(self, h):
        return h if h > 0 else 0

    def derivada(self, h):
        return 1 if h > 0 else 0