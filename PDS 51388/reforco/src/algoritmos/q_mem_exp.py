from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.memorias.memoria_experiencia import MemoriaExperiencia


class QME(QLearning):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, dim_max):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.dim_max = dim_max
        self.memoria_experiencia = MemoriaExperiencia(dim_max)

        self.experiencia = None

    def aprender(self, s, a, r, sn):
        super().aprender(s, a, r, sn)
        self.experiencia = (s, a, r, sn)
        self.memoria_experiencia.actualizar(self.experiencia)
        self.simular()

    def simular(self):
        amostras = self.memoria_experiencia.amostrar(self.num_sim)

        for (s, a, r, sn) in amostras:
            super().aprender((s, a, r, sn))
