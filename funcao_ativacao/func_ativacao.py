from abc import ABC, abstractmethod

"""
Interface das funções de ativação φ (RNA Parte 1 slides 20-26).
Cada função sabe calcular φ(h) e a sua derivada φ'(h) = dφ/dh, que é
precisa na retropropagação (Parte 3 slides 23-24, 38). Assim o neurónio
não precisa de saber qual é a função
chama sempre phi.calcular(h) e phi.derivada(h) e cada classe trata do resto.
"""


class FuncAtivacao(ABC):

    @abstractmethod
    def calcular(self, h):
        pass

    @abstractmethod
    def derivada(self, h):
        pass
