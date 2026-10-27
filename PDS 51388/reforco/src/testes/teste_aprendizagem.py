import random

from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.core.agente import Agente
from reforco.src.core.politica import Politica
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.corredor.pdm_corredor import PDMCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.sel_accao.egreedy import EGreedy


def teste_sel_accao():
    """
    Testa a seleção de ação (Aprendizagem por Reforço Parte 1, slides 17 e 20).

    Accao_sofrega devolve a ação com maior Q(s, a). Com Q 1, 3 e 2 devolve a 1.
    Com valores iguais o shuffle faz com que todas as ações possam sair.

    Seleccionar_accao decide entre aproveitar e explorar com o epsilon. Com
    epsilon 0 só aproveita, saindo sempre a ação sôfrega. Com epsilon 1
    só explora, por isso cada ação sai cerca de um quarto das vezes. Com
    epsilon 0.2 e 4 ações, a fração de escolhas fora da sôfrega é 0.2 * 3 / 4
    igual a 0.15, porque a ação ao acaso também pode ser a sôfrega.
    """

    random.seed(0)

    # Ação 1 tem o maior valor
    memoria = MemoriaAprendizagem()
    memoria.actualizar(0, 0, 1.0)
    memoria.actualizar(0, 1, 3.0)
    memoria.actualizar(0, 2, 2.0)
    sel = EGreedy(memoria, [0, 1, 2], 0)
    print("accao_sofrega com Q 1, 3 e 2 devolve a de maior valor (esperado 1):",
          sel.accao_sofrega(0))

    # Sem valores guardados, todas valem o mesmo
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
    """
    Testa o passo de aprendizagem do SARSA (slides 22 a 25).

    Com Q(s, a) igual a 0, Q(sn, an) igual a 2, r igual a 1, alfa 0.5 e gama
    0.9, o J é 1 + 0.9 * 2 igual a 2.8 e o valor novo é 0 + 0.5 * (2.8 - 0)
    igual a 1.4.

    Quando sn é terminal não há ação seguinte, por isso o J é só a recompensa.
    Com Q(s, a) igual a 0, r igual a 1 e alfa 0.5, o valor novo é
    0 + 0.5 * (1 - 0) igual a 0.5.
    """
    # Q(s, a) = 0, Q(sn, an) = 2, r = 1, alfa = 0.5, gama = 0.9
    memoria = MemoriaAprendizagem()
    memoria.actualizar('sn', 'an', 2.0)
    sarsa = Sarsa(memoria, EGreedy(memoria, ['a'], 0), 0.5, 0.9)
    sarsa.aprender('s', 'a', 1.0, 'sn', 'an')
    print("aprender com Q(s, a) 0, Q(sn, an) 2, r 1, alfa 0.5, gama 0.9 (esperado 1.4):",
          memoria.q('s', 'a'))

    # Sem ação seguinte (an não é passado), sn é terminal
    memoria = MemoriaAprendizagem()
    sarsa = Sarsa(memoria, EGreedy(memoria, ['a'], 0), 0.5, 0.9)
    sarsa.aprender('s', 'a', 1.0, 'sn')
    print("aprender em estado terminal com Q(s, a) 0, r 1, alfa 0.5 (esperado 0.5):",
          memoria.q('s', 'a'))


def teste_corredor():
    """
    Testa o agente com SARSA no corredor e compara com o PDM.

    O agente treina 1000 episódios no ambiente, sem conhecer o modelo, e a
    política sôfrega sai dos Q aprendidos. O PDM calcula a política com o
    modelo conhecido, por programação dinâmica, e serve de referência. As duas
    têm de coincidir nos estados de 2 a 6, onde a melhor ação é DIREITA. Os
    estados 1 e 7 são terminais e não aparecem.
    """
    random.seed(0)

    # O SARSA aprende por experiência no ambiente
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    sel = EGreedy(memoria, ambiente.accoes(), 0.1)
    Agente(ambiente, Sarsa(memoria, sel, 0.1, 0.9)).treinar(1000)
    pi_sarsa = Politica().politica(memoria, ambiente.accoes())

    # O PDM calcula com o modelo conhecido
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
Resultados e conclusões

teste_sel_accao
accao_sofrega com Q 1, 3 e 2 devolveu a ação 1, e com valores iguais saíram as
três ações em 1000 chamadas. Com epsilon 0 saiu sempre a ação 1. Com epsilon 1,
em 10000 escolhas, cada ação saiu entre 2408 e 2550 vezes. Com epsilon 0.2 e 4
ações, a fração fora da sôfrega foi 0.1527.
A ação sôfrega é a de maior Q, e o shuffle faz o desempate ao acaso entre
valores iguais. O epsilon controla a exploração. Com 0 só aproveita e com 1 só
explora, com todas as ações igualmente prováveis. Com 0.2 a fração fora da
sôfrega aproxima-se de 0.2 * 3 / 4 igual a 0.15, porque a ação ao acaso também
pode ser a sôfrega. Os desvios nas contagens são ruído estatístico normal.

teste_sarsa
Com Q(s, a) igual a 0, Q(sn, an) igual a 2, r igual a 1, alfa 0.5 e gama 0.9,
o valor novo foi 1.4. Em estado terminal, com r igual a 1, foi 0.5.
O passo de aprendizagem aproxima Q(s, a) do retorno J, igual a r mais gama
vezes Q(sn, an), na proporção do alfa. Em estado terminal não há ação seguinte,
por isso J é só a recompensa e Q(sn, an) não entra na conta.

teste_corredor
Depois de 1000 episódios, a política do SARSA foi DIREITA nos estados de 2 a 6,
igual à política calculada pelo PDM.
O agente aprendeu o comportamento ótimo só com experiência, sem conhecer T nem
R, e chegou à mesma política que o PDM obtém com o modelo. Os estados 1 e 7 não
aparecem porque são terminais. A política coincide antes de Q convergir, porque
só depende de qual ação tem o maior Q e não do valor exato.

accao_sofrega com Q 1, 3 e 2 devolve a de maior valor (esperado 1): 1
accao_sofrega com valores iguais devolve todas as ações ao acaso (esperado {0, 1, 2}): {0, 1, 2}
seleccionar_accao com epsilon 0 só aproveita (esperado {1}): {1}
seleccionar_accao com epsilon 1 só explora (esperado cerca de 2500 por ação): {0: 2550, 1: 2512, 2: 2530, 3: 2408}
seleccionar_accao com epsilon 0.2 e 4 ações, fração fora da sôfrega (esperado 0.15): 0.1527
aprender com Q(s, a) 0, Q(sn, an) 2, r 1, alfa 0.5, gama 0.9 (esperado 1.4): 1.4
aprender em estado terminal com Q(s, a) 0, r 1, alfa 0.5 (esperado 0.5): 0.5
politica aprendida pelo SARSA em 1000 episódios: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
politica calculada pelo PDM (esperado igual): {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
"""