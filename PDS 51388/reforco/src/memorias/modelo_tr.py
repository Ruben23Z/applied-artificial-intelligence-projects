import random

class ModeloTR:

    def __init__(self):
        self.T = {}
        self.R = {}

    def actualizar(self, s, a, r, sn):
        self.T[(s, a)] = sn  # Atualizar modelo de transição (determinista)
        self.R[(s, a)] = r  # Atualizar modelo de recompensa (determinista)

    def amostrar(self):
        s, a = random.choice(list(self.T.keys()))
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn
