from reforco.src.algoritmos.dyna_q import DynaQ
from reforco.src.algoritmos.dyna_q_prioridade import DynaQPrioridade
from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.algoritmos.q_mem_exp import QME
from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.algoritmos.sarsa_lambda import SarsaLambda
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy

ALGORITMOS = {
    "Q-Learning": lambda mem, sel, alfa, gama, sim, lamb: QLearning(mem, sel, alfa, gama),
    "Dyna-Q": lambda mem, sel, alfa, gama, sim, lamb: DynaQ(mem, sel, alfa, gama, sim),
    "Q-Learning com memória": lambda mem, sel, alfa, gama, sim, lamb: QME(mem, sel, alfa, gama, sim, 200),
    "SARSA": lambda mem, sel, alfa, gama, sim, lamb: Sarsa(mem, sel, alfa, gama),
    "SARSA(lambda)": lambda mem, sel, alfa, gama, sim, lamb: SarsaLambda(mem, sel, alfa, gama, lamb, "acumulacao"),
    "Dyna-Q com prioridade": lambda mem, sel, alfa, gama, sim, lamb: DynaQPrioridade(mem, sel, alfa, gama, sim),
}

# estes algoritmos precisam da ação seguinte para aprender
ON_POLICY = {"SARSA", "SARSA(lambda)"}


def media_movel(valores, janela=20):
    ultimos = valores[-janela:]
    return sum(ultimos) / len(ultimos)


class Treino:
    def __init__(self, nome, accoes, alfa, gama, num_sim, epsilon, lamb=0.9):
        self.nome = nome
        self.alfa = alfa
        self.gama = gama
        self.num_sim = num_sim
        self.lamb = lamb
        self.on_policy = nome in ON_POLICY
        self.mem = MemoriaAprendizagem()
        self.sel = EGreedy(self.mem, accoes, epsilon)
        self.algoritmo = ALGORITMOS[nome](self.mem, self.sel, alfa, gama, num_sim, lamb)
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


    def iniciar_episodio(self):
        """No SARSA(lambda) os rastos não passam de um jogo para o seguinte"""
        rastos = getattr(self.algoritmo, "rastos", None)
        if rastos is not None:
            rastos.limpar()



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
