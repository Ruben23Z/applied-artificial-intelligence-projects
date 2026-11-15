"""
Testes do Q-Learning com memória de experiência (Aprendizagem por Reforço
Parte 5, slides 6, 9 e 10)

teste_aprender
Com r 1, alfa 0.5 e gama 0.9, o passo real dá Q = 0.5. Com num_sim 3, Q passa a
0.75 com uma experiência guardada e a 0.96875 com duas. A memória não passa de dim_max

teste_politica_qme e teste_media_episodios
Treina o QME no corredor, sem PDM, e compara a média de episódios até
Q(4, DIREITA) >= 0.7 com 0 e com 10 recordações por passo
"""
import random

from reforco.src.algoritmos.q_mem_exp import QME
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def novo(num_sim, dim_max, acoes=("a",), epsilon=0):
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, list(acoes), epsilon)
    return memoria, QME(memoria, seletor, 0.5, 0.9, num_sim, dim_max)


def teste_aprender():
    memoria, qme = novo(0, 10)
    qme.aprender('s', 'a', 1.0, 'sn', False)
    print("Q com zero simulações. Esperado: 0.5. Obtido:", memoria.q("s", "a"))

    memoria, qme = novo(3, 10)
    qme.aprender('s', 'a', 1.0, 'sn', False)
    print("Depois da primeira experiência. Esperado: 0.75. Obtido:", memoria.q("s", "a"))
    print("Experiências guardadas. Esperado: [('s', 'a', 1.0, 'sn')]. Obtido:", qme.memoria_experiencia.memoria)

    qme.aprender("s", "a", 1.0, "sn", False)
    print("Depois da segunda experiência. Esperado: 0.96875. Obtido:", memoria.q("s", "a"))

    memoria, qme = novo(1, 2)
    for _ in range(5):
        qme.aprender('s', 'a', 1.0, 'sn', False)
    print("Tamanho da memória após cinco passos. Esperado: 2. Obtido:", len(qme.memoria_experiencia.memoria))

    memoria, qme = novo(3, 10)
    qme.aprender('s', 'a', 1.0, 'sn', True)
    print("Q no caso terminal. Esperado: 0.75. Obtido:", memoria.q("s", "a"))


def episodios_ate_valor(num_sim, semente, alvo=0.7, max_episodios=500):
    random.seed(semente)
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, ambiente.accoes(), 0.1)
    qme = QME(memoria, seletor, 0.1, 0.9, num_sim, 100)

    for numero_episodio in range(1, max_episodios + 1):
        qme.episodio(ambiente)
        if memoria.q(4, AmbienteCorredor.DIREITA) >= alvo:
            return numero_episodio
    return max_episodios


def teste_politica_qme():
    random.seed(0)
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, ambiente.accoes(), 0.1)
    qme = QME(memoria, seletor, 0.1, 0.9, 10, 100)

    for _ in range(100):
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
    total = sum(episodios_ate_valor(numero_simulacoes, semente) for semente in range(numero_sementes))
    return total / numero_sementes


def teste_media_episodios():
    for numero_simulacoes, media_esperada in ((0, 47.41), (10, 8.87)):
        media_obtida = media_episodios_ate_valor(numero_simulacoes)
        print("Simulações por passo:", numero_simulacoes, "| Média esperada:", media_esperada,
              "| Média obtida:", round(media_obtida, 2))


if __name__ == "__main__":
    teste_aprender()
    teste_politica_qme()
    teste_media_episodios()
"""
Resultados e conclusões:

Q com zero simulações. Esperado: 0.5. Obtido: 0.5
Depois da primeira experiência. Esperado: 0.75. Obtido: 0.75
Experiências guardadas. Esperado: [('s', 'a', 1.0, 'sn')]. Obtido: [('s', 'a', 1.0, 'sn')]
Depois da segunda experiência. Esperado: 0.96875. Obtido: 0.96875
Tamanho da memória após cinco passos. Esperado: 2. Obtido: 2
Q no caso terminal. Esperado: 0.75. Obtido: 0.75
Política aprendida pelo QME: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
Simulações por passo: 0 | Média esperada: 47.41 | Média obtida: 47.41
Simulações por passo: 10 | Média esperada: 8.87 | Média obtida: 8.87

A política é DIREITA nos estados de 2 a 6, igual à do PDM. Q(4, DIREITA) chega
a 0.7 em 47.41 episódios sem recordações e em 8.87 com 10, porque a memória
propaga o valor com experiências já vividas, sem modelo.
As recordações por passo são o menor valor entre num_sim e as experiências guardadas
"""
