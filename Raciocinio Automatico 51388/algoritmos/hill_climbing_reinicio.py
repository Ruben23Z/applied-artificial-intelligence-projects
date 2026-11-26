"""
Raciocínio automático Parte 4, slides 25, 28 e 29

O Hill-Climbing é executado várias vezes, cada uma, a partir de um estado inicial
aleatório, até se atingir um objetivo ou o número máximo de reinícios (slide
28). Se nenhuma execução chegar a uma solução, fica-se com o estado de maior
valor entre os que foram obtidos

Existem dois modos de procura, sendo este o modo global, em que a
exploração não depende do estado atual e funciona como uma amostragem do espaço
de estados. Cada reinício sobe numa zona diferente do relevo de valor, e assim
um máximo local deixa de decidir sozinho o resultado. Para isto funcionar, o
estado_inicial do Problema tem de devolver um estado aleatório em cada chamada
Caso contrário todas as execuções seguem o mesmo caminho e reiniciar não adianta
nada
"""

from algoritmos.hill_climbing import HillClimbing


class HillClimbingReinicio:

    def __init__(self):
        """Cria o HillClimbing usado em cada reinício"""
        self.hill_climbing = HillClimbing()

    def iniciar(self, problema, max_reinicios):
        """Corre o Hill-Climbing até max_reinicios vezes e devolve a primeira solução ou o melhor estado obtido"""
        self.melhor_estado = None
        self.valor_melhor_estado = float("-inf")

        for reinicio in range(max_reinicios):
            self.estado = self.hill_climbing.iniciar(problema)

            if problema.solucao(self.estado):
                self.melhor_estado = self.estado
                break
            else:
                self.valor_estado = problema.avaliar(self.estado)

                if self.valor_estado > self.valor_melhor_estado:
                    self.melhor_estado = self.estado
                    self.valor_melhor_estado = self.valor_estado

        return self.melhor_estado
