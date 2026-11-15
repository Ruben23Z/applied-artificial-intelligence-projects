"""
Testes da seleção de ação, do SARSA e do agente no corredor (Aprendizagem por
Reforço Parte 1, slides 17, 20 e 22 a 25)
"""
import random

from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.core.agente import Agente
from reforco.src.core.politica import Politica
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.corredor.pdm_corredor import PDMCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def teste_sel_accao():
    random.seed(0)

    memoria = MemoriaAprendizagem()
    memoria.actualizar(0, 0, 1.0)
    memoria.actualizar(0, 1, 3.0)
    memoria.actualizar(0, 2, 2.0)
    sel = EGreedy(memoria, [0, 1, 2], 0)
    print("accao_sofrega com Q 1, 3 e 2 devolve a de maior valor (esperado 1):",
          sel.accao_sofrega(0))

    sel = EGreedy(MemoriaAprendizagem(), [0, 1, 2], 0)
    escolhidas = set()
    for _ in range(1000):
        escolhidas.add(sel.accao_sofrega(0))
    print("accao_sofrega com valores iguais devolve todas as ações ao acaso (esperado {0, 1, 2}):",
          escolhidas)

    # Daqui para baixo a ação sôfrega é a 1
    memoria = MemoriaAprendizagem()
    memoria.actualizar(0, 1, 1.0)

    sel = EGreedy(memoria, [0, 1, 2], 0)
    escolhidas = set()
    for _ in range(1000):
        escolhidas.add(sel.seleccionar_accao(0))
    print("seleccionar_accao com epsilon 0 só aproveita (esperado {1}):", escolhidas)

    sel = EGreedy(memoria, [0, 1, 2, 3], 1)
    contagem = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(10000):
        contagem[sel.seleccionar_accao(0)] += 1
    print("seleccionar_accao com epsilon 1 só explora (esperado cerca de 2500 por ação):",
          contagem)

    sel = EGreedy(memoria, [0, 1, 2, 3], 0.2)
    fora = 0
    for _ in range(10000):
        if sel.seleccionar_accao(0) != 1:
            fora += 1
    print("seleccionar_accao com epsilon 0.2 e 4 ações, fração fora da sôfrega (esperado 0.15):",
          fora / 10000)


def teste_sarsa():
    memoria = MemoriaAprendizagem()
    memoria.actualizar('sn', 'an', 2.0)
    sarsa = Sarsa(memoria, EGreedy(memoria, ['a'], 0), 0.5, 0.9)
    sarsa.aprender('s', 'a', 1.0, 'sn', 'an')
    print("aprender com Q(s, a) 0, Q(sn, an) 2, r 1, alfa 0.5, gama 0.9 (esperado 1.4):",
          memoria.q('s', 'a'))

    # Sem an, sn é terminal
    memoria = MemoriaAprendizagem()
    sarsa = Sarsa(memoria, EGreedy(memoria, ['a'], 0), 0.5, 0.9)
    sarsa.aprender('s', 'a', 1.0, 'sn')
    print("aprender em estado terminal com Q(s, a) 0, r 1, alfa 0.5 (esperado 0.5):",
          memoria.q('s', 'a'))


def teste_corredor():
    random.seed(0)

    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    sel = EGreedy(memoria, ambiente.accoes(), 0.1)
    Agente(ambiente, Sarsa(memoria, sel, 0.1, 0.9)).treinar(1000)
    pi_sarsa = Politica().politica(memoria, ambiente.accoes())

    pdm = PDMCorredor(0.9)
    pi_pdm = pdm.politica(pdm.utilidade())

    print("politica aprendida pelo SARSA em 1000 episódios:", dict(sorted(pi_sarsa.items())))
    print("politica calculada pelo PDM (esperado igual):",
          {s: pi_pdm[s] for s in sorted(pi_sarsa)})


if __name__ == "__main__":
    teste_sel_accao()
    teste_sarsa()
    teste_corredor()
"""
Resultados e conclusões:

accao_sofrega com Q 1, 3 e 2 devolve a de maior valor (esperado 1): 1
accao_sofrega com valores iguais devolve todas as ações ao acaso (esperado {0, 1, 2}): {0, 1, 2}
seleccionar_accao com epsilon 0 só aproveita (esperado {1}): {1}
seleccionar_accao com epsilon 1 só explora (esperado cerca de 2500 por ação): {0: 2550, 1: 2512, 2: 2530, 3: 2408}
seleccionar_accao com epsilon 0.2 e 4 ações, fração fora da sôfrega (esperado 0.15): 0.1527
aprender com Q(s, a) 0, Q(sn, an) 2, r 1, alfa 0.5, gama 0.9 (esperado 1.4): 1.4
aprender em estado terminal com Q(s, a) 0, r 1, alfa 0.5 (esperado 0.5): 0.5
politica aprendida pelo SARSA em 1000 episódios: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
politica calculada pelo PDM (esperado igual): {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

A política do SARSA coincide com a do PDM sem o agente conhecer T nem R. Coincide
antes de Q convergir, porque só depende de qual ação tem o maior Q. 
Os desvios nas contagens com epsilon 1 são ruído
"""