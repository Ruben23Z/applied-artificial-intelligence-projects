import random

from reforco.src.algoritmos.q_learning import QLearning
from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.core.agente import Agente
from reforco.src.core.politica import Politica
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.corredor.pdm_corredor import PDMCorredor
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
    """
    Testa um episódio do Q-Learning no corredor

    Q(s, DIREITA) = 2 e Q(s, ESQUERDA) = 0 nos estados de 2 a 6(sem ser terminais)
    alfa = 0.5, gama = 0.9, epsilon = 0
    episódio 4, 5, 6, 7

    Q(4, DIREITA) = 2 + 0.5 * (0 + 0.9 * 2 - 2) = 1.9, e escolhe direita, pois é o que apresenta maior ganho
    Q(5, DIREITA) = 1.9, mesmo cálculo
    Q(6, DIREITA) = 2 + 0.5 * (1 - 2) = 1.5, porque 7 é terminal e J = r
    """
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()

    for s in range(2, 7):
        memoria.actualizar(s, AmbienteCorredor.DIREITA, 2.0)
        memoria.actualizar(s, AmbienteCorredor.ESQUERDA, 0.0)

    epsilon = 0
    sel = EGreedy(
        memoria,
        ambiente.accoes(),
        epsilon
    )
    alfa = 0.5
    gama = 0.9

    algoritmo = QLearning(memoria, sel, alfa, gama)
    # O episódio começa no estado 4 e continua até chegar a um terminal
    algoritmo.episodio(ambiente)

    # Recolhe os valores Q da ação DIREITA nos estados 4, 5 e 6
    valores_direita = {}

    for estado in (4, 5, 6):
        valor_q = memoria.q(estado, AmbienteCorredor.DIREITA)
        valor_arredondado = round(valor_q, 3)
        valores_direita[estado] = valor_arredondado

    print("Q(s, DIREITA) depois de um episódio 4, 5, 6, 7 (esperado 4 e 5 com 1.9, 6 com 1.5):",
          valores_direita)

    valor_estado_3 = memoria.q(3, AmbienteCorredor.DIREITA)

    valor_estado_2 = memoria.q(2, AmbienteCorredor.DIREITA)

    print("Q(3, DIREITA), que não mudou, esperado(2):", valor_estado_3)

    print("Q(2, DIREITA), que não mudou, esperado(2):", valor_estado_2)


def teste_sarsa_q_learning():
    """
    Compara o SARSA e o Q-Learning (Parte 3, slides 11 a 13)
    Para treino usa-se 3000 episódios, epsilon = 0.3, alfa = 0.05, gama = 0.9

    Esperado para Q(4, DIREITA), com U(4) = 0.81
    SARSA, abaixo de 0.81, porque aprende o valor da política que executa, com a exploração
    Q-Learning, perto de 0.81, porque aprende o valor da política sôfrega
    """
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

teste_aprender
Q-Learning 1.85, SARSA 0.95 e estado terminal 0.5. O Q-Learning usa o máximo de
Q(sn, a) e o SARSA usa a ação escolhida

teste_episodio
Q(4, DIREITA) e Q(5, DIREITA) ficaram 1.9 e Q(6, DIREITA) 1.5. No passo final
J = r


teste_sarsa_vs_q_learning
SARSA 0.734 e Q-Learning 0.81. O Q-Learning chega a U(4) e o SARSA fica abaixo
por causa da exploração


Q-Learning com Q(sn, a) 1, Q(sn, b) 3, r 1, alfa 0.5, gama 0.9 (esperado 1.85): 1.85
SARSA com os mesmos valores e an igual a a (esperado 0.95): 0.95
Q-Learning em estado terminal com r 1, alfa 0.5 (esperado 0.5): 0.5
Q(s, DIREITA) depois de um episódio 4, 5, 6, 7 (esperado 4 e 5 com 1.9, 6 com 1.5): {4: 1.9, 5: 1.9, 6: 1.5}
Q(3, DIREITA), que não mudou, esperado(2): 2.0
Q(2, DIREITA), que não mudou, esperado(2): 2.0
Q(4, DIREITA) com epsilon 0.3 (esperado SARSA abaixo de 0.81, Q-Learning próximo de 0.81): {'SARSA': 0.734, 'Q-Learning': 0.81}
"""
