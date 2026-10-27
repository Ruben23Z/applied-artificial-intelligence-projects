"""
Memória de aprendizagem (Aprendizagem por Reforço Parte 1, slides 5-9 e 28)

O agente quer saber que ação fazer em cada estado s, para isso precisa do valor
de cada ação, U(s, a), que é o retorno esperado de fazer a no estado s e continuar
a decidir a partir daí (slide 5):

    -> U(s, a) = E[R(s, a, sn) + gama * U(sn, an)]

Com o modelo de transição T(s, a, sn) e o modelo de recompensa R(s, a, sn), o
valor esperado calcula-se com uma soma sobre todos os estados seguintes sn
(slide 6). O agente não tem esses modelos (slide 7). O que tem é experiência,
faz a ação "a" no estado s, vê a recompensa r e o estado seguinte sn. Cada
experiência é uma amostra do valor esperado, e com amostras estima-se o valor
(slide 8):

    U(s, a) ~ Q(s, a)

A memória de aprendizagem é onde o agente guarda as estimativas Q(s, a).

Esta classe define uma memória, onde serve para ler e guardar Q(s, a).
"""


class MemoriaAprendizagem:
    """Tabela Q(s, a) com valor por omissão."""

    def __init__(self, valor_omissao=0.0):
        """Cria a tabela vazia. Os pares sem entrada devolvem valor_omissao."""
        self.valor_omissao = valor_omissao
        self.Q = {}  # Memória de aprendizagem Q: (s, a) -> R
        self.estados = set()  # Estados observados

    def q(self, s, a):
        """Devolve Q(s, a), ou valor_omissao se o par ainda não foi guardado."""
        return self.Q.get((s, a), self.valor_omissao)

    def actualizar(self, s, a, q):
        """Guarda q como Q(s, a) e regista s como estado observado."""
        self.estados.add(s)
        self.Q[(s, a)] = q

    def obter_estados(self):
        """Devolve o conjunto dos estados observados."""
        return self.estados
