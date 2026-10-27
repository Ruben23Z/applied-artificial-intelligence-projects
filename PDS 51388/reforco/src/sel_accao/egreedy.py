import random
from reforco.src.sel_accao.sel_accao import SelAccao

"""
Seleção de ação epsilon-greedy - Aprendizagem por Reforço Parte 1, slides 17 e 20
Em cada passo o agente faz uma de duas escolhas:
    1. com probabilidade 1 - epsilon - > escolhe a ação sôfrega
    2. com probabilidade epsilon-> explora e escolhe uma ação ao acaso

Com epsilon igual a 0 o agente só aproveita os caminhos que tem maior valor de utlidade, bom para quando o modelo já 
está totalmente treinado.
Com epsilon igual a 1 só explora. O valor de epsilon não muda ao longo da aprendizagem.
A ação ao acaso é escolhida entre todas as ações, por isso pode coincidir com a
ação sôfrega.
"""


class EGreedy(SelAccao):
    """Escolhe ações com probabilidade fixa de exploração."""

    def __init__(self, mem_aprend, accoes, epsilon):
        """
        Recebe a memória de aprendizagem, a lista das ações e a probabilidade
        epsilon de explorar, entre 0 e 1.
        """
        super().__init__(mem_aprend, accoes)
        self.epsilon = epsilon

    def seleccionar_accao(self, s):
        """Aproveita com a probabilidade restante e explora com epsilon."""
        chance = random.random()
        if chance > self.epsilon:
            accao = self.accao_sofrega(s)  # Aproveitar conhecimento
        else:
            accao = self.accao_exploraccao()  # Explorar
        return accao
