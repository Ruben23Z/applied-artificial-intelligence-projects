"""
Agente, Processos de Decisão Sequencial slide 4 e Aprendizagem por Reforço
Parte 1 slide 24.

O agente executa uma ação a no ambiente, observa o estado seguinte sn e a
recompensa r, e aprende com essa experiência. Não conhece o algoritmo de
aprendizagem. Em cada episódio entrega o ambiente ao controlo, que corre o
ciclo do algoritmo no método episodio. No SARSA a ação an escolhe-se antes de
aprender. No Q-Learning, no Dyna-Q e no QME a ação de aprendizagem é a sofrega
e obtém-se dentro de aprender. Assim o mesmo Agente serve todos os algoritmos.
"""


class Agente:
    """Liga o ambiente ao método de aprendizagem"""

    def __init__(self, ambiente, controlo):
        """
        Recebe o ambiente e o método de aprendizagem, por exemplo, um Sarsa
        A estratégia de seleção de ação está em controlo.sel_accao.
        controlo é o método de aprendizagem, por exemplo, um Sarsa
        """
        self.ambiente = ambiente
        self.controlo = controlo

    def treinar(self, num_episodios):
        for _ in range(num_episodios):
            self.controlo.episodio(self.ambiente)  # um episódio do algoritmo

