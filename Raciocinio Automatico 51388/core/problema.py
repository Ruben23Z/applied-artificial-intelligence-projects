"""
Raciocínio automático Parte 4, slides 2, 7 a 10 e 18, e Parte 5, slides 8 a 10

Um problema de otimização consiste em encontrar o estado que maximiza uma função
de valor ou minimiza uma função de perda (slide 2). A função de avaliação define
um relevo sobre o espaço de estados e o objetivo é chegar ao ponto mais alto
desse relevo. Em muitos problemas o espaço é tão grande que a procura
exaustiva deixa de ser viável, e é por isso que se recorre à procura local
(slide 8). O treino da rede neuronal da primeira parte do projeto é um caso
destes, porque o estado são os pesos da rede e o que se quer é minimizar o erro
quadrático médio.

Na Parte 5 o problema é o elemento que guarda a informação
própria do domínio, como as distâncias entre as cidades no caixeiro viajante, e
que define o estado inicial, a transição e a avaliação. Aqui a transição é feita
pelo método sucessores, e os sucessores devem resultar de transformações locais,
por exemplo, mover uma rainha na sua coluna ou alterar um troço do percurso no
caixeiro viajante. Se forem gerados ao acaso, a procura passa a ser quase uma
amostragem global e o desempenho piora. Na avaliação, um valor maior corresponde
a um estado melhor. Nos problemas com custo usa-se o custo com valor negativo,
tal como no exemplo das rainhas, onde um estado com 9 conflitos vale menos 9
"""

from abc import ABC, abstractmethod


class Problema(ABC):

    @abstractmethod
    def estado_inicial(self):
        """Devolve o estado onde a procura começa."""

    @abstractmethod
    def sucessores(self, estado):
        """Devolve a lista de estados sucessores do estado dado"""

    @abstractmethod
    def avaliar(self, estado):
        """Devolve o valor do estado, quanto maior melhos"""

    def solucao(self, estado):
        """Indica se o estado é uma solução. Por omissão nenhum estado é"""
        return False

    def sucessor_aleatorio(self, estado):
        """Devolve um único sucessor do estado, escolhido ao acaso"""
        raise NotImplementedError
