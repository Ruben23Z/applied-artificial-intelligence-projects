"""
Ambiente de teste, Processos de Decisão Sequencial slide 12, processo de decisão de markov.

Tem sete posições numa fila.
Os estados são as posições de 1 a 7. As ações são ESQUERDA e DIREITA, e cada uma
move o agente uma posição. As transições são deterministas. A recompensa é -1 ao
chegar ao estado 1, +1 ao chegar ao estado 7 e 0 nos outros passos. Os estados 1
e 7 são terminais. O episódio começa no estado 4, que é o estado médio.

O agente tem acesso por interação, só se pode chamar iniciar() e executar(a), que devolve uma amostra(sn, r, terminou).
O agente não vê as probabilidades nem as recompensas, só o que acontece, o que o SARSA usa com aprendizagem
por reforço com o modelo desconhecido"""


class AmbienteCorredor:
    """Interface entre o agente e o problema a resolver."""
    ESQUERDA = -1
    DIREITA = 1

    def __init__(self):
        self.s = 4

    def iniciar(self):
        """Põe o agente no estado 4 e devolve o estado."""
        self.s = 4
        return self.s

    def executar(self, a):
        """Move o agente e devolve o estado seguinte, a recompensa e se terminou."""
        self.s += a
        if self.s == 7:
            r = 1
        elif self.s == 1:
            r = -1
        else:
            r = 0

        if self.s == 1 or self.s == 7:
            terminal = True
        else:
            terminal = False

        return self.s, r, terminal

    def accoes(self):
        """Devolve a lista das ações possíveis."""
        return [self.ESQUERDA, self.DIREITA]
