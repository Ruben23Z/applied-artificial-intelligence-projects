"""
Memória de experiência, Aprendizagem por Reforço Parte 5, slides 3, 4 e 10

Guarda as últimas dim_max experiências do agente, com cada tuplo
(s, a, r, sn). Com a memória cheia, atualizar remove a mais antiga antes
de juntar a nova. O amostrar devolve até max_amostras experiências
escolhidas ao acaso, com distribuição uniforme e sem repetição

Esta guarda os dados a partir dos quais essas estimativas
voltam-se a atualizar, e é também o conjunto de exemplos de treino quando Q
é representado por uma rede neuronal (slide 3). Difere do ModeloTR da
Parte 4, que guarda um só resultado por par (s, a), porque aqui as
experiências ficam por ordem, com os pares repetidos. O limite dim_max
responde à necessidade de limitar a dimensão (slide 8)
"""
import random


class MemoriaExperiencia:
    def __init__(self, dim_max):
        self.dim_max = dim_max
        self.memoria = []

    def actualizar(self, experiencia):
        """
        Guarda a experiência e remove a mais antiga se a memória estiver cheia
        Args:
            experiencia: tuplo (s, a, r, sn)
        """
        if len(self.memoria) == self.dim_max:
            self.memoria.pop(0)
        self.memoria.append(experiencia)

    def amostrar(self, max_amostras):
        """
        Devolve experiências diferentes escolhidas ao acaso
        """
        num_amostras = min(max_amostras, len(self.memoria))
        return random.sample(self.memoria, num_amostras)

