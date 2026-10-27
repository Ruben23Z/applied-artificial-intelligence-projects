"""
Política comportamental determinista (Processos de Decisão Sequencial,
slides 19, 20 e 34).

Uma política representa o comportamento do agente: define que ação realizar
em cada estado. Esta é determinista, ou seja, indica uma só
ação por estado, em vez de uma probabilidade por par (estado, ação).

A política é sôfrega e não explora. Permite ver o que o agente aprendeu,
não para o treinar.
"""

class Politica:

    def politica(self, mem_aprend, accoes):
        """
        Devolve um dicionário que associa cada estado observado à ação com maior
        Q(s, a). Os estados terminais não aparecem, porque o agente nunca age neles.
        """
        pi = {}
        for s in mem_aprend.obter_estados():
            if accoes:
                melhor_accao = None
                melhor_valor = None
                for a in accoes:
                    valor = mem_aprend.q(s, a)
                    if melhor_valor is None or valor > melhor_valor:
                        melhor_valor = valor
                        melhor_accao = a
                pi[s] = melhor_accao
        return pi