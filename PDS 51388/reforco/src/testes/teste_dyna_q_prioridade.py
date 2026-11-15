import random

from reforco.src.algoritmos.dyna_q import DynaQ
from reforco.src.algoritmos.dyna_q_prioridade import DynaQPrioridade
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.fila_prioridade import FilaPrioridade
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def novo(num_sim, limiar=0.0):
    memoria = MemoriaAprendizagem()
    return memoria, DynaQPrioridade(memoria, EGreedy(memoria, ['a'], 0), 0.5, 0.9, num_sim, limiar)


def teste_fila():
    fila = FilaPrioridade()
    fila.inserir('x', 0.2)
    fila.inserir('y', 0.9)
    fila.inserir('z', 0.5)
    fila.inserir('x', 0.7)
    saidas = []
    while not fila.vazia():
        saidas.append(fila.remover())
    print("Ordem da fila. Esperado: ['y', 'x', 'z']. Obtido:", saidas)


def teste_aprender():
    memoria, dq = novo(0)
    dq.aprender('s', 'a', 0.0, 'sn', False)
    print("Fila com r 0. Esperado: True. Obtido:", dq.fila.vazia())
    dq.aprender('s', 'a', 1.0, 'sn', False)
    print("Fila com r 1. Esperado: 1. Obtido:", len(dq.fila.prioridades))

    memoria, dq = novo(1)
    dq.aprender('s1', 'a', 0.0, 's2', False)
    dq.aprender('s2', 'a', 1.0, 't', True)
    print("num_sim 1 Q(s2, a). Esperado: 0.75. Obtido:", memoria.q('s2', 'a'))
    print("num_sim 1 Q(s1, a). Esperado: 0.0. Obtido:", memoria.q('s1', 'a'))
    print("Prioridade de (s1, a). Esperado: 0.675. Obtido:", dq.fila.prioridades.get(('s1', 'a')))

    memoria, dq = novo(2)
    dq.aprender('s1', 'a', 0.0, 's2', False)
    dq.aprender('s2', 'a', 1.0, 't', True)
    print("num_sim 2 Q(s1, a). Esperado: 0.3375. Obtido:", memoria.q('s1', 'a'))


def episodios_ate_valor(fabrica, semente, alvo=0.7, max_episodios=500):
    random.seed(semente)
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    algoritmo = fabrica(memoria, EGreedy(memoria, ambiente.accoes(), 0.1))
    for n in range(1, max_episodios + 1):
        algoritmo.episodio(ambiente)
        if memoria.q(4, AmbienteCorredor.DIREITA) >= alvo:
            return n
    return max_episodios


def teste_corredor(numero_sementes=50):
    casos = (("Dyna-Q", lambda m, s: DynaQ(m, s, 0.1, 0.9, 3)),
             ("Dyna-Q com prioridade", lambda m, s: DynaQPrioridade(m, s, 0.1, 0.9, 3)))
    for nome, fabrica in casos:
        total = sum(episodios_ate_valor(fabrica, k) for k in range(numero_sementes))
        print(nome, "| média de episódios:", round(total / numero_sementes, 2))


if __name__ == "__main__":
    teste_fila()
    teste_aprender()
    teste_corredor()

"""
Resultados e conclusões:

Ordem da fila. Esperado: ['y', 'x', 'z']. Obtido: ['y', 'x', 'z']
Fila com r 0. Esperado: True. Obtido: True
Fila com r 1. Esperado: 1. Obtido: 1
num_sim 1 Q(s2, a). Esperado: 0.75. Obtido: 0.75
num_sim 1 Q(s1, a). Esperado: 0.0. Obtido: 0.0
Prioridade de (s1, a). Esperado: 0.675. Obtido: 0.675
num_sim 2 Q(s1, a). Esperado: 0.3375. Obtido: 0.3375
Dyna-Q | média de episódios: 22.14
Dyna-Q com prioridade | média de episódios: 26.84



A fila devolve a maior prioridade primeiro e guarda cada par uma só vez, com a maior.
Com num_sim 1 só o par terminal é atualizado e o predecessor (s1, a) fica
na fila com prioridade 0.675, e com num_sim 2 o valor propaga-se para trás e
Q(s1, a) chega a 0.3375.
No corredor o Dyna-Q com prioridade precisou de mais
episódios (26.84) do que o Dyna-Q (22.14), ao contrário do esperado
O corredor tem poucos estados e o Q(4, DIREITA) depende da propagação desde o estado 7
O Dyna-Q simula pares ao acaso, que são sempre úteis num espaço tão pequeno, e a
prioridade limita-se a ordenar poucos pares """