"""
Raciocínio automático Parte 5, slides 2 a 7

É uma versão estocástica do Hill-Climbing em que também se aceitam transições
para estados com menos valor do que o atual, onde em cada passo escolhe-se
um sucessor ao acaso.
Se o valor aumentar, a transição é aceite, se diminuir, só
é aceite com a probabilidade e^(delta_E / T),
em que delta_E é a diferença entre o valor do sucessor e o do estado atual
e T é a temperatura (slide 4).
Como delta_E é negativo, quanto maior a perda e mais baixa a temperatura, menor é
essa probabilidade. Por aceitar às vezes estados piores, o algoritmo consegue
sair de um máximo local

A temperatura é dada pela função de arrefecimento f_t(função de escalonamento),
e vai descendo ao longo do tempo.
O t_max limita o tempo da procura, por isso a solução devolvida é aproximada e pode não ser ótima(slide 2)

No exemplo do caixeiro viajante a temperatura começa em T(0) = 100 e, em cada
passo, é multiplicada por k = 0.99, ou seja, T(t) = T(0) * k^t
Assim vai descendo aos poucos e, quanto mais baixa fica,
menos vezes o algoritmo aceita um percurso mais longo do que o atual

A temperatura funciona como o epsilon(e) da aprendizagem por reforço, quando está
alta o algoritmo explora mais, e quando está baixa aproveita o que já explorou.
"""

import math
import random


class SimulatedAnnealing:

    def iniciar(self, problema, f_t, t_max):
        """Aplica o Simulated Annealing de t igual a 0 até t_max e devolve o estado final
        Args:
            problema: problema de otimização a resolver.
            f_t: função de arrefecimento que dá a temperatura T no instante t.
            t_max: último instante de tempo, que limita o número de iterações.
        """
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)
        for t in range(t_max + 1):
            T = f_t(t)
            sucessores = problema.sucessores(self.estado)
            sucessor = random.choice(sucessores)

            if problema.solucao(sucessor):
                self.estado = sucessor
                break

            valor_sucessor = problema.avaliar(sucessor)
            delta_E = valor_sucessor - self.valor_estado
            if delta_E > 0 or random.random() < math.exp(delta_E / T):
                self.estado = sucessor
                self.valor_estado = valor_sucessor

        return self.estado
