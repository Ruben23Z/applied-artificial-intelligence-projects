"""
Q-Learning com memória de experiência, Aprendizagem por Reforço Parte 5,
slides 5 a 10.

Acrescenta ao Q-Learning a recordação de experiências passadas. A cada
paço real o agente atualiza Q(s, a), guarda a experiência na
MemoriaExperiencia e repete num_sim atualizações com amostras dessa
memória, com a fórmula do Q-Learning. Os passos recordados não exigem
interação com o ambiente e aceleram a propagação de valor (slide 7).

Tem o mesmo objetivo do Dyna-Q da Parte 4, mas a origem das amostras é
diferente. O Dyna-Q simula a partir de um modelo aprendido e o QME
reutiliza experiências que aconteceram, sem modelo. Por isso, num
ambiente não determinista, o QME recorda resultados diferentes do mesmo
par (s, a), enquanto o ModeloTR guarda só o último.

Tal como no Dyna-Q, as amostras não guardam final (slide 9). Num estado
terminal Q(sn, a) mantém o valor por omissão da memória, por isso com 0.0
o J fica igual a r
"""
from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.memorias.memoria_experiencia import MemoriaExperiencia


class QME(QLearning):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, dim_max):
        """Q-Learning com memória de experiencia"""
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.dim_max = dim_max
        self.memoria_experiencia = MemoriaExperiencia(dim_max)

        self.experiencia = None

    def aprender(self, s, a, r, sn, final):
        """
        Atualiza Q e guarda a experiência na memória
        Args:
           s: estado
           a: ação feita em s
           r: recompensa recebida
           sn: estado seguinte
           final: se sn é terminal
        """
        super().aprender(s, a, r, sn, final)
        self.experiencia = (s, a, r, sn)
        self.memoria_experiencia.actualizar(self.experiencia)
        self.simular()

    def simular(self):
        amostras = self.memoria_experiencia.amostrar(self.num_sim)

        for (s, a, r, sn) in amostras:
            super().aprender(s, a, r, sn)
