"""
Processos de Decisão Sequencial, slides 11 e 32 a 34.

PDM com modelo conhecido, definido por cinco componentes:
    - estados, o conjunto S
    - accoes(s), as ações A(s) possíveis em s
    - T(s, a, sn), probabilidade de ir de s para sn com a ação a
    - R(s, a, sn), recompensa de ir de s para sn com a ação a
    - gama, o fator de desconto
    - suc(s) devolve os estados seguintes possíveis a partir de s.

Uaccao(s, a, U) = soma sobre sn em suc(s) de T(s, a, sn) * (R(s, a, sn) + gama * U(sn))
A utilidade de um estado é a da melhor ação. Calcula-se por iteração, sempre
sobre uma cópia Uant, e para quando delta <= delta_max, em que delta é a maior
diferença entre U e Uant. A política é a ação de maior Uaccao em cada estado.
Os estados terminais não têm ações, ficam com U = 0 e não entram na política.
"""


class PDM:
    """Modelo do mundo e cálculo da utilidade e da política."""

    def __init__(self, estados, accoes, T, R, suc, gama, delta_max=1e-6):
        """
        Recebe a lista dos estados, as funções accoes(s), T(s, a, sn), R(s, a, sn)
        e suc(s), e o fator de desconto gama.
        """
        self.estados = estados
        self.accoes = accoes
        self.T = T
        self.R = R
        self.suc = suc
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
                accoes = self.accoes(s)
                if accoes:  # terminais ficam com U = 0
                    U[s] = max(self.Uaccao(s, a, Uant) for a in accoes)
                    delta = max(delta, abs(U[s] - Uant[s]))
            if delta <= self.delta_max:
                return U

    def Uaccao(self, s, a, U):
        """Devolve a utilidade da ação a no estado s, dado o dicionário U."""
        soma = sum(self.T(s, a, sn) * (self.R(s, a, sn) + self.gama * U[sn]) for sn in self.Ssucessivo(s))

        return soma

    def politica(self, U):
        """
        Devolve um dicionário estado -> ação de maior Uaccao(s, a, U).
        Os estados terminais (sem ações) não entram
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
