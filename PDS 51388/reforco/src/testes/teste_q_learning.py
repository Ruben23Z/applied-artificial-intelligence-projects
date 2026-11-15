"""
Testes do Q-Learning (Aprendizagem por Reforço Parte 3, slides 6 a 13)

teste_aprender
Com Q(sn, a) 1, Q(sn, b) 3, r 1, alfa 0.5 e gama 0.9, o Q-Learning usa o máximo
de Q(sn, a'), J = 1 + 0.9 * 3 = 3.7 e Q(s, a) = 1.85. O SARSA com an = a dá
J = 1.9 e Q(s, a) = 0.95. Em estado terminal J = r e Q(s, a) = 0.5

teste_episodio
Com Q(s, DIREITA) 2 e Q(s, ESQUERDA) 0 nos estados de 2 a 6, o episódio
4, 5, 6, 7 dá Q(4) = Q(5) = 2 + 0.5 * (0.9 * 2 - 2) = 1.9 e Q(6) = 2 + 0.5 * (1 - 2)
= 1.5, porque 7 é terminal e J = r. Os estados 2 e 3 não mudam

teste_sarsa_q_learning
Treina 3000 episódios com epsilon 0.3. O Q-Learning aprende o valor da política
sôfrega e aproxima-se de U(4) = 0.81. O SARSA aprende o valor da política que
executa, com exploração, e fica abaixo
"""

import random

from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.core.agente import Agente
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def teste_aprender():
    """
    Testa o passo de aprendizagem
    Q(sn, a) = 1, Q(sn, b) = 3, r = 1, alfa = 0.5, gama = 0.9
    J = 1 + 0.9 * 3 = 3.7
    Q(s, a) = 0 + 0.5 * (3.7 - 0) = 1.85

    Para comparação com SARSA com an = a:
        J = 1 + 0.9 * 1 = 1.9
        Q(s, a) = 0.95
    """
    memoria = MemoriaAprendizagem()

    memoria.actualizar('sn', 'a', 1.0)
    memoria.actualizar('sn', 'b', 3.0)
    eg = EGreedy(memoria, ['a', 'b'], 0)

    ql = QLearning(memoria, eg, 0.5, 0.9)

    ql.aprender('s', 'a', 1.0, 'sn', False)
    print("Q-Learning com Q(sn, a) 1, Q(sn, b) 3, r 1, alfa 0.5, gama 0.9 (esperado 1.85):",
          memoria.q('s', 'a'))

    # SARSA
    memoria = MemoriaAprendizagem()
    memoria.actualizar('sn', 'a', 1.0)
    memoria.actualizar('sn', 'b', 3.0)
    eg2 = EGreedy(memoria, ['a', 'b'], 0)
    sarsa = Sarsa(memoria, eg2, 0.5, 0.9)
    sarsa.aprender('s', 'a', 1.0, 'sn', 'a')
    print("SARSA com os mesmos valores e an igual a a (esperado 0.95):", memoria.q('s', 'a'))

    memoria = MemoriaAprendizagem()
    memoria.actualizar('sn', 'b', 3.0)
    eg3 = EGreedy(memoria, ['a', 'b'], 0)
    ql = QLearning(memoria, eg3, 0.5, 0.9)
    ql.aprender('s', 'a', 1.0, 'sn', True)
    print("Q-Learning em estado terminal com r 1, alfa 0.5 (esperado 0.5):", memoria.q('s', 'a'))



def teste_episodio():
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()

    for s in range(2, 7):
        memoria.actualizar(s, AmbienteCorredor.DIREITA, 2.0)
        memoria.actualizar(s, AmbienteCorredor.ESQUERDA, 0.0)

    sel = EGreedy(memoria, ambiente.accoes(), 0)
    algoritmo = QLearning(memoria, sel, 0.5, 0.9)
    algoritmo.episodio(ambiente)

    valores_direita = {estado: round(memoria.q(estado, AmbienteCorredor.DIREITA), 3)
                       for estado in (4, 5, 6)}
    print("Q(s, DIREITA) depois de um episódio 4, 5, 6, 7 (esperado 4 e 5 com 1.9, 6 com 1.5):",
          valores_direita)
    print("Q(3, DIREITA), que não mudou, esperado(2):", memoria.q(3, AmbienteCorredor.DIREITA))
    print("Q(2, DIREITA), que não mudou, esperado(2):", memoria.q(2, AmbienteCorredor.DIREITA))


def teste_sarsa_q_learning():
    resultados = {}
    for nome, algoritmo in (("SARSA", Sarsa), ("Q-Learning", QLearning)):
        random.seed(0)
        ambiente = AmbienteCorredor()
        memoria = MemoriaAprendizagem()
        sel = EGreedy(memoria, ambiente.accoes(), 0.3)
        Agente(ambiente, algoritmo(memoria, sel, 0.05, 0.9)).treinar(3000)
        resultados[nome] = round(memoria.q(4, AmbienteCorredor.DIREITA), 3)

    print("Q(4, DIREITA) com epsilon 0.3 (esperado SARSA abaixo de 0.81, Q-Learning próximo de 0.81):",
          resultados)


if __name__ == "__main__":
    teste_aprender()
    teste_episodio()
    teste_sarsa_q_learning()

"""
Resultados e conclusões

Q-Learning com Q(sn, a) 1, Q(sn, b) 3, r 1, alfa 0.5, gama 0.9 (esperado 1.85): 1.85
SARSA com os mesmos valores e an igual a a (esperado 0.95): 0.95
Q-Learning em estado terminal com r 1, alfa 0.5 (esperado 0.5): 0.5
Q(s, DIREITA) depois de um episódio 4, 5, 6, 7 (esperado 4 e 5 com 1.9, 6 com 1.5): {4: 1.9, 5: 1.9, 6: 1.5}
Q(3, DIREITA), que não mudou, esperado(2): 2.0
Q(2, DIREITA), que não mudou, esperado(2): 2.0
Q(4, DIREITA) com epsilon 0.3 (esperado SARSA abaixo de 0.81, Q-Learning próximo de 0.81): {'SARSA': 0.734, 'Q-Learning': 0.81}

O Q-Learning usa o máximo de Q(sn, a') e o SARSA a ação escolhida, por isso o
Q-Learning chega a U(4) = 0.81 e o SARSA fica abaixo (0.734) por causa da exploração.
No passo final J = r
"""
