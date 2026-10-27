import random
from abc import ABC, abstractmethod

"""
Seleção de ação - Aprendizagem por Reforço Parte 1, slides 15-17 e 20

Em cada estado o agente escolhe uma ação. O que sabe sobre o valor das ações
está na memória de aprendizagem, em Q(s, a). Essas estimativas vêm da
experiência, só existem para o que o agente já experimentou e mudam à medida
que experimenta mais.

Há duas maneiras de escolher (slide 15):
    1. Aproveitamento-> faz a ação com maior valor estimado, a ação sôfrega(greedy)
        a_t = a*
    2. Exploração-> fazer uma ação diferente da sôfrega, para ganhar experiência e corrigir as estimativas
        a_t != a_t*

Só aproveitar deixa o agente a repetir as ações que estimou com mais valor
no início, sem testar as outras. Só explorar faz com que nunca use o que
aprendeu. É o dilema de explorar ou aproveitar (slide 17).

Cada estratégia de seleção responde a esta pergunta de maneira diferente.
Esta classe tem o que é comum a todas, escolher a ação sôfrega e escolher uma
ação de exploração.
"""


class SelAccao(ABC):
    """Base das estratégias de seleção de ação."""

    def __init__(self, mem_aprend, accoes):
        """
         Recebe a memória de aprendizagem onde se lê Q(s, a) e a lista das ações
        que o agente pode fazer. A lista é a mesma em todos os estados.

        Args:
            mem_aprend: memória de aprendizagem de onde se lê Q(s, a)
            accoes: lista das ações que o agente pode fazer. A lista é a mesma
                em todos os estados
        """
        self.mem_aprend = mem_aprend
        self.accoes = accoes

    def accao_sofrega(self, s):
        """
        Devolve a ação com maior Q(s, a) no estado s (slide 20). Se várias
        ações têm o mesmo valor, escolhe uma delas ao acaso. Sem isto o agente
        escolhia sempre a primeira da lista, e no início, com todos os valores
        iguais ao valor por omissão, nunca experimentava as outras.

           Args:
               s: estado
           Returns:
               A ação sôfrega.
       """

        melhor_valor = None
        melhores_accoes = []

        for a in self.accoes:
            valor = self.mem_aprend.q(s, a)

            if melhor_valor is None or valor > melhor_valor:
                melhor_valor = valor
                melhores_accoes = [a]
            elif valor == melhor_valor:
                melhores_accoes.append(a)

        return random.choice(melhores_accoes)

    def accao_exploraccao(self):
        """
             Devolve uma ação ao acaso entre todas, todas com a mesma probabilidade
             (slide 20). Pode devolver a ação sôfrega.
             Returns:
                 Uma ação.
         """
        return random.choice(self.accoes)

    @abstractmethod
    def seleccionar_accao(self, s):
        """Decide entre aproveitar e explorar e devolve a ação a executar."""
