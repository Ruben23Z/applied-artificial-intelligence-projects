from reforco.src.algoritmos.dyna_q import DynaQ
from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.algoritmos.q_mem_exp import QME
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy

ALGORITMOS = {
    "Q-Learning": lambda mem, sel, alfa, gama, sim: QLearning(mem, sel, alfa, gama),
    "Dyna-Q": lambda mem, sel, alfa, gama, sim: DynaQ(mem, sel, alfa, gama, sim),
    "Q-Learning com memória": lambda mem, sel, alfa, gama, sim: QME(mem, sel, alfa, gama, sim, 200),
}


def media_movel(valores, janela=20):
    ultimos = valores[-janela:]
    return sum(ultimos) / len(ultimos)


class Treino:
    def __init__(self, nome, accoes, alfa, gama, num_sim, epsilon):
        self.nome = nome
        self.alfa = alfa
        self.gama = gama
        self.num_sim = num_sim
        self.mem = MemoriaAprendizagem()
        self.sel = EGreedy(self.mem, accoes, epsilon)
        self.algoritmo = ALGORITMOS[nome](self.mem, self.sel, alfa, gama, num_sim)
        self.episodios = 0
        self.passos = 0
        self.melhor = 0
        self.recompensa_episodio = 0
        self.pontuacoes = []
        self.recompensas = []
        self.epsilons = []
        self.estados = []
        self.medias = []
        self.medias_recompensa = []

    def registar(self, pontuacao):
        recorde = pontuacao > self.melhor
        self.episodios += 1
        self.melhor = max(self.melhor, pontuacao)
        self.pontuacoes.append(pontuacao)
        self.recompensas.append(self.recompensa_episodio)
        self.epsilons.append(self.sel.epsilon)
        self.estados.append(len(self.mem.obter_estados()))
        self.medias.append(media_movel(self.pontuacoes))
        self.medias_recompensa.append(media_movel(self.recompensas))
        return recorde
