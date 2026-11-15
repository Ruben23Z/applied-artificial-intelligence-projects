"""
Algoritmo Q-Learning, Aprendizagem por Reforço Parte 3, slides 6 a 9.

Tal como o SARSA da Parte 1, estima Q(s, a) por amostragem, sem conhecer
os modelos T e R, com a atualização por diferença temporal
    -> Q(s, a) = Q(s, a) + alfa * (J - Q(s, a))

A diferença está no retorno J. No SARSA, an é a ação que o agente executa
 a seguir, podendo ser de exploração (on-policy).
No Q-Learning, an é a ação sôfrega em sn, obtida com accao_sofrega, e
    J = r + gama * max Q(sn, a')
O agente executa ações com exploração, mas aprende o valor da política
sofrega (off-policy). Isto aproxima a atualização da equação de Bellman
dos Processos de Decisão Sequencial, em que U(s) é o máximo sobre as
ações, agora estimado por amostras e não calculado com T e R. Q converge
para Q* se cada par (s, a) for visitado um número ilimitado de vezes e
alfa tender para 0 (slide 15).
Em estado terminal não há ação seguinte e J é igual a r, onde O parâmetro final indica isso.
"""
from reforco.src.algoritmos.sarsa import Sarsa


class QLearning(Sarsa):
    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        super().__init__(mem_aprend, sel_accao, alfa, gama)

    def aprender(self, s, a, r, sn, final=False):
        """
        Atualiza Q(s, a) com an sôfrega em sn
        """
        if final:
            an = None
        else:
            an = self.sel_accao.accao_sofrega(sn)

        super().aprender(s, a, r, sn, an)

    def episodio(self, ambiente):
        """
        Corre um episódio até um estado terminal
        """

        s = ambiente.iniciar()
        terminou = False

        while not terminou:
            a = self.sel_accao.seleccionar_accao(s)
            sn, r, terminou = ambiente.executar(a)
            self.aprender(s, a, r, sn, terminou)
            s = sn
