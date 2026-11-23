import math
import random


class SimulatedAnnealing:


    def iniciar(self, problema, f_t,t_max):
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)
        for t in range(t_max):
            T = f_t(t)
            sucessores = problema.sucessores(self.estado)
            sucessor = random.choice(sucessores)

            if problema.solucao(sucessor):
                self.estado = sucessor
                break

            valor_sucessor = problema.avaliar(sucessor)
            delta_E = valor_sucessor - self.valor_estado
            if delta_E > 0 or random < math.exp(delta_E / T):
                self.estado = sucessor
                self.valor_estado = valor_sucessor
        return self.estado
