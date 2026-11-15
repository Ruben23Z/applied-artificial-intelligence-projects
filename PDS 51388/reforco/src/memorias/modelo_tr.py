"""
Modelo do mundo do Dyna-Q, Aprendizagem por Reforço Parte 4, slide 22.

Guarda a transição T[(s, a)] = sn e a recompensa R[(s, a)] = r da última
vez que o agente fez a em s. Corresponde às funções T(s, a, sn) e
R(s, a, sn) dos Processos de Decisão Sequencial, mas construídas a partir
da experiência e não definidas à partida (Parte 1, modelos desconhecidos)

Assume ambiente determinista, por isso cada par guarda um só resultado.
Num ambiente não determinista, T teria de ser uma distribuição de
probabilidade

O método actualizar regista a experiência real e amostrar devolve um par
(s, a) já experimentado, escolhido ao acaso, com o sn e o r guardados.
Só se simulam pares vividos, porque o modelo não tem informação sobre os
outros.
"""
import random


class ModeloTR:

    def __init__(self):
        self.T = {}
        self.R = {}

    def actualizar(self, s, a, r, sn):
        """
        Guarda sn e r para o par (s, a)
        Args:
            s: estado
            a: ação feita em s
            r: recompensa recebida
            sn: estado seguinte
        """
        self.T[(s, a)] = sn  # Atualizar modelo de transição (determinista)
        self.R[(s, a)] = r  # Atualizar modelo de recompensa (determinista)

    def amostrar(self):
        """Devolve (s, a, r, sn) de um par já guardado, escolhido ao acaso"""
        s, a = random.choice(list(self.T.keys()))
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn

