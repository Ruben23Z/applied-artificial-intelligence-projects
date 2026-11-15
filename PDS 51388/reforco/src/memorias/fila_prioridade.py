"""
Fila de prioridade do varrimento com prioridade, Parte 4, slides 18 e 19

Cada par (s, a) tem uma prioridade igual à variação do valor que a
atualização provocaria, quanto maior a variação mais informação nova. Guarda-se
num dicionário par -> prioridade, por isso um par só conta uma vez, com a maior
prioridade.
"""


class FilaPrioridade:

    def __init__(self):
        self.prioridades = {}  # par -> prioridade em vigor

    def inserir(self, par, p):
        if p > self.prioridades.get(par, 0):
            self.prioridades[par] = p

    def remover(self):
        """Devolve o par de maior prioridade, ou None se a fila estiver vazia"""
        melhor_par = None
        melhor_prioridade = None
        for par, p in self.prioridades.items():
            if melhor_prioridade is None or p > melhor_prioridade:
                melhor_par = par
                melhor_prioridade = p
        if melhor_par is not None:
            del self.prioridades[melhor_par]
        return melhor_par

    def vazia(self):
        return len(self.prioridades) == 0
