import random

from reforco.src.algoritmos.dyna_q import DynaQ
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def teste_aprender():
    """Testa o passo do Dyna-Q (Parte 4, slides 14 e 21)
    """
    epsilon = 0
    alfa = 0.5
    gama = 0.9
    numero_simulacoes = 0
    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, ["a"], epsilon)

    dyna = DynaQ(memoria, seletor, alfa, gama, numero_simulacoes)
    dyna.aprender('s', 'a', 1.0, 'sn', False)
    print("Q com zero simulações. Esperado: 0.5. Obtido:", memoria.q("s", "a"))

    estado_seguinte_guardado = dyna.modelo.T[("s", "a")]
    recompensa_guardada = dyna.modelo.R[("s", "a")]

    print("Estado seguinte guardado. Esperado: sn. Obtido:", estado_seguinte_guardado)
    print("Recompensa guardada. Esperado: 1.0. Obtida:", recompensa_guardada)

    # Cada caso começa com uma memória nova
    # Assim, os casos não interferem entre si
    casos_de_simulacao = ((1, 0.75), (2, 0.875), (3, 0.9375))

    for numero_simulacoes, valor_esperado in casos_de_simulacao:
        memoria = MemoriaAprendizagem()
        seletor = EGreedy(memoria, ["a"], epsilon)

        dyna = DynaQ(memoria, seletor, alfa, gama, numero_simulacoes)
        dyna.aprender('s', 'a', 1.0, 'sn', False)
        valor_q = memoria.q("s", "a")

        print("Número de simulações:", numero_simulacoes, "| Q esperado:", valor_esperado, "| Q obtido:", valor_q)

    memoria = MemoriaAprendizagem()
    seletor = EGreedy(memoria, ["a"], epsilon)

    dyna = DynaQ(memoria, seletor, alfa, gama, 3)

    dyna.aprender('s', 'a', 1.0, 'sn', True)
    valor_q_terminal = memoria.q("s", "a")

    print("Q no caso terminal com três simulações.", "Esperado: 0.9375. Obtido:", valor_q_terminal)


def episodios_ate_valor(num_sim, semente, alvo=0.7, max_episodios=500):
    random.seed(semente)

    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    acoes = ambiente.accoes()
    epsilon = 0.1
    seletor = EGreedy(memoria, acoes, epsilon)

    alfa = 0.1
    gama = 0.9

    dyna = DynaQ(memoria, seletor, alfa, gama, num_sim)

    for numero_episodio in range(1, max_episodios + 1):
        dyna.episodio(ambiente)

        valor_q = memoria.q(4, AmbienteCorredor.DIREITA)

        if valor_q >= alvo:
            return numero_episodio

    return None


def teste_dyna_q_no_corredor():
    """
    Compara o treino sem simulações e com três simulações
    Este teste usa o ambiente corredor, mas não usa o PDM
    """

    configuracoes = (0, 3)

    for numero_simulacoes in configuracoes:
        numero_episodios = episodios_ate_valor(numero_simulacoes, 0)

        if numero_episodios is None:
            print("Com", numero_simulacoes, "simulações, o alvo não foi atingido.")
        else:
            print("Com", numero_simulacoes, "simulações, o alvo foi atingido ao fim de", numero_episodios, "episódios.")


if __name__ == "__main__":
    teste_aprender()
    teste_dyna_q_no_corredor()
"""
Conclusões

teste_aprender
Q foi 0.5 com num_sim 0 e 0.75, 0.875 e 0.9375 com 1, 2 e 3. 
Em estado terminal o resultado é igual, porque o modelo não guarda final e Q(sn, a) vale 0


Resultados:
Q com zero simulações. Esperado: 0.5. Obtido: 0.5
Estado seguinte guardado. Esperado: sn. Obtido: sn
Recompensa guardada. Esperado: 1.0. Obtida: 1.0
Número de simulações: 1 | Q esperado: 0.75 | Q obtido: 0.75
Número de simulações: 2 | Q esperado: 0.875 | Q obtido: 0.875
Número de simulações: 3 | Q esperado: 0.9375 | Q obtido: 0.9375
Q no caso terminal com três simulações. Esperado: 0.9375. Obtido: 0.9375
Com 0 simulações, o alvo foi atingido ao fim de 46 episódios
Com 3 simulações, o alvo foi atingido ao fim de 19 episódios
"""