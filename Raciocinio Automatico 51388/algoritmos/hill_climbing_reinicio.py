from algoritmos.hill_climbing import HillClimbing


class HillClimbingReinicio:

    def __init__(self, max_reinicios=1000):
        self.max_tentativas = max_reinicios
        self.hill_climbing = HillClimbing()

    def iniciar(self, problema):
        self.melhor_estado = None
        self.valor_melhor_estado = float("-inf")

        for reinicio in range(self.max_tentativas):
            self.estado = self.hill_climbing.iniciar(problema)

            if problema.solucao(self.estado):
                self.melhore_estado = self.estado
                break
            else:
                self.valor_estado = problema.avaliar(self.estado)

                if self.valor_estado > self.valor_melhor_estado:
                    self.melhor_estado = self.estado
                    self.valor_melhor_estado = self.valor_estado

        return self.melhore_estado
