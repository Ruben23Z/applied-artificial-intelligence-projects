from funcao_ativacao.func_ativacao import FuncAtivacao

"""
Função degrau (RNA Parte 1 slide 24): produz 1 se h atinge o limiar (0) e
um valor "falso" caso contrário (0 em codificação binária, -1 em bipolar).
Não é diferenciável, por isso a derivada devolve 0 e só serve para propagação
(neurónios com pesos fixos, funções lógicas), não treina por retropropagação.
estrito=True usa h > 0 em vez de h >= 0 (condição do exemplo REDE XOR, Parte 2 slide 23).
"""


class Degrau(FuncAtivacao):

    def __init__(self, falso=0, estrito=False):
        self.falso = falso
        self.estrito = estrito

    def calcular(self, h):
        ativo = h > 0 if self.estrito else h >= 0
        return 1 if ativo else self.falso

    def derivada(self, h):
        return 0
