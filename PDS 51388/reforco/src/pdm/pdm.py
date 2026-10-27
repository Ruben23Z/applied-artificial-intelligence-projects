"""
Processos de Decisão Sequencial slides 11 e 32 a 34- Processo de decisão de Markov

Um PDM tem cinco componentes:
    1. estados, o conjunto S dos estados
    2. accoes(s), as ações A(s) possíveis no estado s
    3. T(s, a, sn), a probabilidade de passar de s para sn ao fazer a
    4. R(s, a, sn), a recompensa por passar de s para sn ao fazer a gama, o fator de desconto

O método suc(s) devolve os estados seguintes possíveis a partir de s.
A utilidade de uma ação é o valor esperado do que se ganha ao fazê-la.
Uaccao(s, a, U) = soma sobre sn em suc(s) de T(s, a, sn) * (R(s, a, sn) + gama * U(sn))
A utilidade de um estado é a da melhor ação. Começa-se com U(s) igual a 0 e
repete-se o cálculo, sempre sobre uma cópia do U anterior, até o erro delta ser
menor ou igual a delta_max. O erro é a maior diferença entre o U novo e o anterior.
A política é a ação de maior utilidade em cada estado.
_-----------------------------------------------------------------------------


Política comportamental determinista (Processos de Decisão Sequencial,
slides 19, 20 e 34).

Uma política representa o comportamento do agente, que define que ação realizar
em cada estado. Esta é determinista, ou seja, indica uma só
ação por estado, em vez de uma probabilidade por par (estado, ação).

A política ótima escolhe em cada estado a ação de maior utilidade (slide 34):
    -> pi(s) = argmax sobre a em A(s) de Uaccao(s, a, U)

Em PDM, Uaccao calcula-se com os modelos T e R. Na aprendizagem por
reforço esses modelos não existem, por isso o valor da ação é a estimativa
Q(s, a) guardada na memória de aprendizagem
"""


class PDM:
    """Modelo do mundo e cálculo da utilidade e da política."""

    def __init__(self, estados, accoes, T, R, Ssucessivo, gama, delta_max=1e-6):
        """
        Recebe a lista dos estados, as funções accoes(s), T(s, a, sn), R(s, a, sn)
        e Ssucessivo(s), e o fator de desconto gama.
        """
        self.estados = estados
        self.accoes = accoes
        self.T = T
        self.R = R
        self.Ssucessivo = Ssucessivo
        self.gama = gama
        self.delta_max = delta_max

    def utilidade(self):
        """Devolve o dicionário U com a utilidade de cada estado."""
        U = {}
        S = self.estados

        for s in S:
            U[s] = 0

        while True:
            Uant = U.copy()
            delta = 0
            for s in S:
                valores = []
                for a in self.accoes(s):
                    valores.append(self.Uaccao(s, a, Uant))
                U[s] = max(valores)
                delta = max(delta, abs(U[s] - Uant[s]))
            if delta <= self.delta_max:
                return U

    def Uaccao(self, s, a, U):
        """Devolve a utilidade da ação a no estado s, dado o dicionário U."""
        soma = sum(self.T(s, a, sn) * (self.R(s, a, sn) + self.gama * U[sn]) for sn in self.Ssucessivo(s))

        return soma

    def politica(self, U):
        """
        Devolve um dicionário que associa cada estado observado à ação com maior
        Q(s, a). Os estados terminais não aparecem, porque o agente nunca age neles.

        """
        pi = {}
        for s in self.estados:
            if self.accoes(s):
                melhor_accao = None
                melhor_valor = None
                for a in self.accoes(s):
                    valor = self.Uaccao(s, a, U)
                    if melhor_valor is None or valor > melhor_valor:
                        melhor_valor = valor
                        melhor_accao = a
                pi[s] = melhor_accao
        return pi
