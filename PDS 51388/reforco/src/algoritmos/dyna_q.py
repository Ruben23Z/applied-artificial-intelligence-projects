from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.memorias.modelo_tr import ModeloTR


class DynaQ(QLearning):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()

    def aprender(self, s, a, r, sn): #DUVIDA PROFFFFFFFFFFFFFFFFFFFFFFFF
        super().aprender(s, a, r, sn)  # Actualizar Q(s,a)
        self.modelo.actualizar(s, a, r, sn)  # Actualizar modelo
        self.simular()  # Realizar simulação intern

    def simular(self):
        for _ in range(self.num_sim):  # Simular passos de aprendizagem com base no modelo
            s, a, r, sn = self.modelo.amostrar()
            super().aprender(s, a, r, sn)
