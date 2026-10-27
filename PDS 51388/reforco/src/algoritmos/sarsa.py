"""
Algoritmo SARSA, Aprendizagem por Reforço Parte 1, slides 8-14 e 22-25.

O agente não conhece o modelo do ambiente, estimando o valor de cada ação
a partir da experiência. Cada passo dá uma amostra com a sequência(s, a, r, sn, an)
em que s é o estado, a é a ação feita em s, r a recompensa recebida, sn o estado
seguinte e an a ação futura escolhida em sn.

A aprendizagem com política de seleção de ação unica (on-policy).
A estratégia que escolhe as ações
a executar é a mesma que serve para atualizar Q. A ação an usada na atualização
é a que o agente vai mesmo executar a seguir, escolhida com a estratégia de
seleção, por exemplo, epsilon-greedy. Como a estratégia tem exploração, an pode
ser uma ação de exploração.


O J é o retorno da amostra, é o valor que a experiência atual sugere para
Q(s, a), e é com ele que a estimativa antiga é comparada, dependendo das recompensas futuras
Se sn não é terminal, o J junta a recompensa recebida com o valor estimado do
que segue, descontado pelo gama(o fator de desconto).
    -> J(s,a) = r + gama * Q(sn, an)
Se sn é terminal, o episódio acaba e não há ação seguinte nem recompensas
depois, por isso o J é só a recompensa observada (slide 25).
    -> J(s,a) = r

A diferença entre o J e a estimativa antiga é o erro de estimativa. O alfa é o
fator de aprendizagem e decide que parte desse erro entra em Q(s, a).
Perto de 0 pesam mais as recompensas antigas e perto de 1 pesam mais as recentes.
    -> Q(s, a) = Q(s, a) + αlfa * (r + γQ(s’, a’) – Q(s, a))
    -> Q(s, a) = Q(s, a) + alfa * (J - Q(s, a))
"""


class Sarsa:
    """Aprendizagem por reforço SARSA."""

    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        """
        Recebe a memória de aprendizagem, a estratégia de seleção de ação, o
        fator de aprendizagem alfa e o fator de desconto gama, ambos entre 0 e 1.
        """
        self.mem_aprend = mem_aprend
        self.sel_accao = sel_accao
        self.alfa = alfa
        self.gama = gama

    def aprender(self, s, a, r, sn, an=None):
        """
        Lê Q(s, a) e Q(s', a') na memória, calcula o valor novo de Q(s, a) e guarda-o
        Args:
            s: estado
            a: ação feita em s
            r: recompensa recebida
            sn: estado seguinte
            an: ação escolhida em sn. None se sn é terminal
        """
        q = self.mem_aprend.q(s, a)
        if an is None:  # Verifica se o próximo estado é terminal
            j = r
        else:
            q_sn = self.mem_aprend.q(sn, an)
            j = r + self.gama * q_sn

        Qn = q + self.alfa * (j - q)
        self.mem_aprend.actualizar(s, a, Qn)
