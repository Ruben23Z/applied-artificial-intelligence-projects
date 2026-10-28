

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

        for _ in range(num_episodios):
            self.controlo.episodio(self.ambiente)  # um episódio do algoritmo
