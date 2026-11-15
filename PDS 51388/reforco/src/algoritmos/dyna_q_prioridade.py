"""
Algoritmo Dyna-Q com prioridade (prioritized sweeping), Aprendizagem por
Reforço Parte 4, slides 18 a 20

O Dyna-Q escolhe ao acaso os pares do modelo a simular. Aqui simulam-se
primeiro os pares com maior variação de valor, medida por
    P = |r + gama * max Q(sn, an) - Q(s, a)|
e só entram na fila se P for maior que o limiar. Depois de atualizar um par,
calcula-se a prioridade de todos os pares que levam a esse estado (atualização
retrospetiva), varrendo os encadeamentos das transições para trás

Herda do QLearning e não do DynaQ, porque o aprender do DynaQ já simula e
simular-se-ia dentro das simulações.
"""
from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.memorias.fila_prioridade import FilaPrioridade
from reforco.src.memorias.modelo_tr_pred import ModeloTRPred


class DynaQPrioridade(QLearning):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, limiar=0.0):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.limiar = limiar
        self.modelo = ModeloTRPred()
        self.fila = FilaPrioridade()
        self.actualizacoes = 0  # atualizações de Q, serve para o gráfico do slide 20

    def prioridade(self, s, a, r, sn):
        q_max = max(self.mem_aprend.q(sn, an) for an in self.sel_accao.accoes)
        return abs(r + self.gama * q_max - self.mem_aprend.q(s, a))

    def aprender(self, s, a, r, sn, final=False):
        """Calcula a prioridade antes de atualizar"""
        self.modelo.actualizar(s, a, r, sn)
        p = self.prioridade(s, a, r, sn)
        if p > self.limiar:
            self.fila.inserir((s, a), p)
        super().aprender(s, a, r, sn, final)
        self.actualizacoes += 1
        self.simular()

    def simular(self):
        for _ in range(self.num_sim):
            par = self.fila.remover()
            if par is None:
                break
            s, a = par
            r, sn = self.modelo.R[par], self.modelo.T[par]
            super().aprender(s, a, r, sn)
            self.actualizacoes += 1
            for sp, ap in self.modelo.predecessores(s):
                rp = self.modelo.R[(sp, ap)]
                p = self.prioridade(sp, ap, rp, s)
                if p > self.limiar:
                    self.fila.inserir((sp, ap), p)