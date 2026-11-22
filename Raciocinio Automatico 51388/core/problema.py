from abc import ABC, abstractmethod


class Problema(ABC):

    @abstractmethod
    def estado_inicial(self):
        """Devolve um estado inicial"""

    @abstractmethod
    def sucessores(self, estado):
        """Devolve os estados vizinhos do estado dado"""

    @abstractmethod
    def avaliar(self, estado):
        """Devolve o valor do estado, quanto maior melhor"""

    def solucao(self, estado):
        """Diz se o estado é uma solução, por omissão nenhum estado é"""
        return False

    @abstractmethod
    def sucessor_aleatorio(self, estado):
        """Devolve um único estado vizinho escolhido ao acaso"""