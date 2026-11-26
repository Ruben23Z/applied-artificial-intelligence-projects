"""
Raciocínio automático Parte 4, slides 11 a 14 e 20 a 22

O Hill-Climbing é um método de procura local. Os métodos exaustivos
percorrem o espaço de estados de forma sistemática, ao passo que aqui
só se exploram os estados adjacentes ao atual (slide 11). A procura começa no
estado inicial e, em cada iteração, gera os sucessores, que resultam de
transformações incrementais, e passa para o estado que tiver melhor avaliação.
Quando nenhum sucessor tem valor superior ao do estado atual, a procura para e
devolve esse estado

Como só mantém o estado atual e os seus sucessores, a complexidade espacial é
linear no fator de ramificação e a temporal também é linear na dimensão da
solução (slide 20).
Em contrapartida, o algoritmo não é completo nem ótimo,
porque pode ficar preso num máximo local, num local plano ou num
planalto, onde nenhum vizinho é melhor
Guardar os estados visitados resolveria a questão,
mas a memória cresceria de forma exponencial (slide 22).
"""


class HillClimbing:

    def iniciar(self, problema):
        """Sobe pelo melhor sucessor até nenhum melhorar o valor e devolve o estado final"""
        self.estado = problema.estado_inicial()
        # O valor inicial é menos infinito para que a primeira transição seja sempre aceite, e se o estado não tiver
        # sucessores o ciclo termina logo.
        self.valor_estado = float("-inf")
        while True:
            self.sucessores = problema.sucessores(self.estado)
            self.novo_estado = max(self.sucessores, key=problema.avaliar, default=None)
            if self.novo_estado is None:
                break

            self.valor_novo_estado = problema.avaliar(self.novo_estado)
            if self.valor_novo_estado <= self.valor_estado:
                break

            self.estado = self.novo_estado
            self.valor_estado = self.valor_novo_estado

        return self.estado
