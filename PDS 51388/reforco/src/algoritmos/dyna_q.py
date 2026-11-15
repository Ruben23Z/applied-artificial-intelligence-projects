"""
Algoritmo Dyna-Q, Aprendizagem por Reforço Parte 4, slides 13 a 17 e 21

Junta a aprendizagem direta, sendo o passo do Q-Learning, com
planeamento a partir de um modelo do mundo aprendido. A cada paço real o
agente atualiza Q(s, a), regista a transição no ModeloTR e repete num_sim
passos de aprendizagem com transições tiradas desse modelo, sem
interação com o ambiente.

O modelo é a versão aprendida das funções T e R que os Processos de
Decisão Sequencial assumem conhecidas. Com num_sim igual a 0 o
comportamento é o do Q-Learning. Com num_sim maior, cada experiência real
propaga valor várias vezes, reduzindo o número de episódios necessários(slide 17)

Tem o mesmo objetivo do QME, que é complementar a experiência
real com atualizações adicionais, mas as amostras vêm dum modelo
aprendido e não das experiências guardadas.
Num estado terminal Q(sn, a) mantém o valor por omissão da memória, por isso com 0.0
o J fica igual a r.
"""
from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.memorias.modelo_tr import ModeloTR


class DynaQ(QLearning):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()

    def aprender(self, s, a, r, sn, final):
        """
        Atualiza Q, regista a transição no modelo e simula
        Args:
            s: estado
            a: ação feita em s
            r: recompensa recebida
            sn: estado seguinte
            final: se sn é terminal
        """

        super().aprender(s, a, r, sn, final)  # Actualizar Q(s,a)
        self.modelo.actualizar(s, a, r, sn)  # Actualizar modelo
        self.simular()  # Realizar simulação interna

    def simular(self):
        """Repete num_sim passos de aprendizagem com transições do modelo"""
        for _ in range(self.num_sim):  # Simular passos de aprendizagem com base no modelo
            s, a, r, sn = self.modelo.amostrar()
            super().aprender(s, a, r, sn)
