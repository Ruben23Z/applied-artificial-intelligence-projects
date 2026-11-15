import random

from reforco.src.algoritmos.q_mem_exp import QME
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def teste_aprender():
    """
    Testa o passo do Q-Learning com memória de experiência (Parte 5, slides 6,
    9 e 10)

    Com Q(sn, a) 0, r 1, alfa 0.5 e gama 0.9, o paço real dá Q(s, a) = 0.5 e
    guarda a experiência. O amostrar não repete nem passa das experiências
    guardadas, por isso com num_sim 3 e uma experiência há uma recordação e Q
    passa a 0.75. À segunda vez há duas, e Q passa a 0.96875. A memória não
    passa de dim_max
    """

    alfa = 0.5
    gama = 0.9
    acoes = ["a"]
    epsilon = 0

    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, acoes, epsilon)
    # Sem simulações
    numero_simulacoes = 0
    limite_memoria = 10

    qme = QME(memoria, seletor, alfa, gama, numero_simulacoes, limite_memoria)
    qme.aprender('s', 'a', 1.0, 'sn', False)
    valor_q = memoria.q("s", "a")
    print("Q com zero simulações Esperado: 0.5 Obtido:", valor_q)

    # Com 3 simulações
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, acoes, epsilon)
    numero_simulacoes = 3
    limite_memoria = 10

    qme = QME(memoria, seletor, alfa, gama, numero_simulacoes, limite_memoria)

    # só existe uma experiência para recordar
    qme.aprender('s', 'a', 1.0, 'sn', False)
    valor_q = memoria.q("s", "a")
    print("Depois da primeira experiência. Esperado: 0.75. Obtido:", valor_q)

    print("Experiências guardadas. Esperado: [('s', 'a', 1.0, 'sn')]. Obtido:", qme.memoria_experiencia.memoria)

    # agora existem duas experiências para recordar.
    qme.aprender("s", "a", 1.0, "sn", False)

    valor_q = memoria.q("s", "a")
    print("Depois da segunda experiência. Esperado: 0.96875. Obtido:", valor_q)

    # limitar a memória a duas experiências.
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, acoes, epsilon)
    numero_simulacoes = 1
    limite_memoria = 2

    qme = QME(memoria, seletor, alfa, gama, numero_simulacoes, limite_memoria)
    for passo in range(5):
        qme.aprender('s', 'a', 1.0, 'sn', False)
    tamanho_memoria = len(qme.memoria_experiencia.memoria)
    print("Tamanho da memória após cinco passos. Esperado: 2. Obtido:", tamanho_memoria)

    # teste para o estado seguinte ser terminal
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, acoes, epsilon)
    numero_simulacoes = 3
    limite_memoria = 10

    qme = QME(memoria, seletor, alfa, gama, numero_simulacoes, limite_memoria)
    qme.aprender('s', 'a', 1.0, 'sn', True)
    valor_q_terminal = memoria.q("s", "a")
    print("Q no caso terminal. Esperado: 0.75. Obtido:", valor_q_terminal)


def episodios_ate_valor(num_sim, semente, alvo=0.7, max_episodios=500):
    """
    Devolve quantos episódios o QME precisou para atingir o valor alvo
    """

    random.seed(semente)

    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    acoes = ambiente.accoes()
    epsilon = 0.1
    seletor = EGreedy(memoria, acoes, epsilon)

    alfa = 0.1
    gama = 0.9
    limite_memoria = 100

    qme = QME(memoria, seletor, alfa, gama, num_sim, limite_memoria)

    for numero_episodio in range(1, max_episodios + 1):
        qme.episodio(ambiente)

        valor_q = memoria.q(4, AmbienteCorredor.DIREITA)

        if valor_q >= alvo:
            return numero_episodio

    # Devolve o limite se o alvo não for atingido.
    return max_episodios


def teste_politica_qme():
    """
    Treina o QME no corredor e mostra a ação que aprendeu em cada estado.
    Não usa o PDM
    """

    random.seed(0)

    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    acoes = ambiente.accoes()
    seletor = EGreedy(memoria, acoes, 0.1)

    alfa = 0.1
    gama = 0.9
    numero_simulacoes = 10
    limite_memoria = 100

    qme = QME(memoria, seletor, alfa, gama, numero_simulacoes, limite_memoria)

    numero_episodios = 100

    for episodio in range(numero_episodios):
        qme.episodio(ambiente)

    politica_aprendida = {}

    for estado in sorted(memoria.obter_estados()):
        valor_esquerda = memoria.q(estado, AmbienteCorredor.ESQUERDA)

        valor_direita = memoria.q(estado, AmbienteCorredor.DIREITA)

        if valor_direita >= valor_esquerda:
            politica_aprendida[estado] = AmbienteCorredor.DIREITA
        else:
            politica_aprendida[estado] = AmbienteCorredor.ESQUERDA

    print("Política aprendida pelo QME:", politica_aprendida)


def media_episodios_ate_valor(numero_simulacoes, numero_sementes=100):
    """
    Calcula a média de episódios necessários em várias execuções
    """

    total_episodios = 0

    for semente in range(numero_sementes):
        episodios = episodios_ate_valor(numero_simulacoes, semente)

        total_episodios = total_episodios + episodios

    media = total_episodios / numero_sementes
    return media


def teste_media_episodios():
    """
    Compara o treino sem recordações e com dez recordações por passo
    """

    casos = (
        (0, 47.41),
        (10, 8.87)
    )

    for numero_simulacoes, media_esperada in casos:
        media_obtida = media_episodios_ate_valor(numero_simulacoes)

        print("Simulações por passo:", numero_simulacoes, "| Média esperada:", media_esperada, "| Média obtida:",
              round(media_obtida, 2))


if __name__ == "__main__":
    teste_aprender()
    teste_politica_qme()
    teste_media_episodios()
"""
Resultados

teste_aprender
Q foi 0.5 com num_sim 0, 0.75 com uma experiência e 0.96875 à segunda vez. Com
dim_max 2 a memória ficou com 2 experiências. As recordações por passo são o
menor valor entre num_sim e as experiências guardadas

teste_corredor
A política foi DIREITA nos estados de 2 a 6, igual à do PDM. Q(4, DIREITA)
chegou a 0.7 em 47.41 episódios com num_sim 0 e em 8.87 com 10. A memória
propaga o valor com experiências já vividas, sem modelo



Resultados

Q com zero simulações Esperado: 0.5 Obtido: 0.5
Depois da primeira experiência. Esperado: 0.75. Obtido: 0.75
Experiências guardadas. Esperado: [('s', 'a', 1.0, 'sn')]. Obtido: [('s', 'a', 1.0, 'sn')]
Depois da segunda experiência. Esperado: 0.96875. Obtido: 0.96875
Tamanho da memória após cinco passos. Esperado: 2. Obtido: 2
Q no caso terminal. Esperado: 0.75. Obtido: 0.75
Política aprendida pelo QME: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
Simulações por passo: 0 | Média esperada: 47.41 | Média obtida: 47.41
Simulações por passo: 10 | Média esperada: 8.87 | Média obtida: 8.87
"""
