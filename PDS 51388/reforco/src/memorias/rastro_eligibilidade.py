"""
Rasto de elegibilidade, Aprendizagem por Reforço Parte 4, slides 5 e 6

Guarda e(s, a) para cada par visitado no episódio. Sobe na visita e decai com
gama * lambda a cada passo. O slide 5 define dois tipos
    acumulação ->    e = gama * lambda * e + 1 se o par é visitado
    substituição -> e = 1 se o par é visitado
Nos outros casos e = gama * lambda * e. Na acumulação o rasto pode passar de 1,
na substituição não. É a estrutura de apoio do SarsaLambda, como o ModeloTR é
para o DynaQ, e não substitui a MemoriaAprendizagem, que guarda Q
"""


class RastroEligibilidade:

    def __init__(self, modo):
        if modo not in ('acumulacao', 'substituicao'):
            raise ValueError(f"Modo desconhecido: {modo}")
        self.modo = modo
        self.e = {}  # (s, a) -> rasto

    def visitar(self, s, a):
        if self.modo == 'acumulacao':
            self.e[(s, a)] = self.e.get((s, a), 0) + 1
        else:
            self.e[(s, a)] = 1

    def decair(self, fator):
        for par in self.e:
            self.e[par] = fator * self.e[par]

    def pares(self):
        return list(self.e.items())

    def limpar(self):
        self.e.clear()
