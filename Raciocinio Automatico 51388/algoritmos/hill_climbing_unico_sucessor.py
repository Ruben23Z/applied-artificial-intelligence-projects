"""
Raciocínio automático Parte 4, slide 27

Nesta variante os sucessores são gerados um a um, ao acaso, até aparecer um que
aumente o valor do estado atual. Esse estado é aceite logo, sem olhar para os
restantes. Compensa quando cada estado tem um número muito elevado de
sucessores, porque gerá-los e avaliá-los todos em cada iteração precisaria de
muito processamento.
Se o número de tentativas atingir o máximo sem se detetar melhoria,
considera-se que o estado atual é um máximo e a procura acaba
"""


class HillClimbingUnicoSucessor:

    def __init__(self, max_tentativas=1000):
        self.max_tentativas = max_tentativas

    def iniciar(self, problema):
        """Aceita o primeiro sucessor aleatório que melhore o estado, até esgotar as tentativas, e devolve o
        estado final"""
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)

        while True:
            melhorou = False
            for _ in range(self.max_tentativas):
                sucessor = problema.sucessor_aleatorio(self.estado)
                self.valor = problema.avaliar(sucessor)
                if self.valor > self.valor_estado:
                    self.estado = sucessor
                    self.valor_estado = self.valor
                    melhorou = True
                    break

            if not melhorou:
                break

        return self.estado
