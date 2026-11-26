"""
Raciocínio automático Parte 4, slides 24 e 26

É uma variante do Hill-Climbing em que o próximo estado deixa de ser sempre o
melhor sucessor. O algoritmo avalia todos os sucessores, guarda os que aumentam
o valor do estado atual e escolhe um deles ao acaso (slide 26). Um estado pior
nunca é aceite, por isso a aleatoriedade só entra na escolha entre melhorias

Isto aparece como uma forma de controlar a exploração e o
aproveitamento. Aproveitar é seguir a opção sôfrega, a transição que maximiza o
valor local, e explorar é escolher com um critério aleatório. A mesma ideia já
tinha surgido na aprendizagem por reforço com a política epsilon-greedy. A
convergência é mais lenta do que no Hill-Climbing base, mas consoante a
topologia do espaço de estados pode chegar a soluções melhores.
"""

import random


class HillClimbingEstocastico:

    def iniciar(self, problema):
        """Sobe por melhorias escolhidas ao acaso até nenhum sucessor aumentar o valor e devolve o estado final"""
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)

        while True:
            melhores = []
            for sucessor in problema.sucessores(self.estado):
                self.valor = problema.avaliar(sucessor)
                if self.valor > self.valor_estado:
                    melhores.append((sucessor, self.valor))

            if not melhores:
                break

            self.estado, self.valor_estado = random.choice(melhores)

        return self.estado
