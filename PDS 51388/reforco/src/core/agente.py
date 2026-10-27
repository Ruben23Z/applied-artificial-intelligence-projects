"""
Agente, Processos de Decisão Sequencial slide 4 e Aprendizagem por Reforço
Parte 1 slide 24.

O agente executa uma ação "a" no ambiente, observa o estado seguinte sn e a
recompensa r, e aprende, depois o agente atua.

O algoritmo SARSA percorre cada episódio assim:
    1. Inicia o ambiente e obtém o estado s.
    2. Escolhe a ação a em s com a estratégia de seleção.
    3. Executa a e observa r e sn.
    4. Escolhe an em sn com a estratégia da seleção.
    5. Aprende com (s, a, r, sn, an).
    6. Passa a ter s igual a sn e a igual a an.
    7. Repete a partir do passo 3 até sn ser um estado terminal.
"""


class Agente:
    """Liga o ambiente ao método de aprendizagem."""

    def __init__(self, ambiente, controlo):
        """
        Recebe o ambiente e o método de aprendizagem, por exemplo, um Sarsa. A
        estratégia de seleção de ação está em controlo.sel_accao.

        Args:
            ambiente: ambiente com o qual o agente interage.
            controlo: Método de aprendizagem, por exemplo, um Sarsa.
        """
        self.ambiente = ambiente
        self.controlo = controlo

    def treinar(self, num_episodios):
        """Corre num_episodios episódios e aprende em cada passo
        Args:
             num_episodios: número de episódios a executar.
        """
        sel_accao = self.controlo.sel_accao
        for _ in range(num_episodios):
            s = self.ambiente.iniciar()
            a = sel_accao.seleccionar_accao(s)

            terminou = False
            while not terminou:
                sn, r, terminou = self.ambiente.executar(a)
                if terminou:
                    an = None
                else:
                    an = sel_accao.seleccionar_accao(sn)

                self.controlo.aprender(s, a, r, sn, an)
                s = sn
                a = an
