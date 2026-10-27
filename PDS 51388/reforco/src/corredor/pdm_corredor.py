from reforco.src.pdm.pdm import PDM
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor

"""
PDM de teste, Processos de Decisão Sequencial slide 12.

É o mesmo corredor do AmbienteCorredor, agora com o modelo conhecido. Os estados
são de 1 a 7 e as ações são ESQUERDA e DIREITA. A transição é determinista nos
estados de 2 a 6, e os estados 1 e 7 não têm estados seguintes. A recompensa é 1
ao chegar ao estado 7, -1 ao chegar ao estado 1 e 0 nos outros casos.

Descreve o mesmo corredor de testes com o modelo explícito T(s, a, sn), R(s, a, sn), suc(s), accoes(s) e gama. 
Com isso o PDM calcula a utilidade e a politica diretamente.
"""


class PDMCorredor(PDM):

    def __init__(self, gama=0.9):
        super().__init__([1, 2, 3, 4, 5, 6, 7],
                         self._accoes, self._T, self._R, self._suc, gama)

    def _accoes(self, s):
        return [AmbienteCorredor.ESQUERDA, AmbienteCorredor.DIREITA]

    def _T(self, s, a, sn):
        return 1 if s in (2, 3, 4, 5, 6) and sn == s + a else 0

    def _R(self, s, a, sn):
        if sn == 7:
            return 1
        if sn == 1:
            return -1
        return 0

    def _suc(self, s):
        return [] if s in (1, 7) else [s - 1, s + 1]
