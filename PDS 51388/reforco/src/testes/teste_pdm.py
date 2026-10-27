"""
Testa o PDM do corredor com fator de desconto 0.9 (Processos de Decisão
Sequencial, slides 32 a 34). O modelo é conhecido, por isso tudo é calculado
sem simular o ambiente.

Uaccao(s, a, U) é o valor esperado de fazer a ação a no estado s. Com U a 0,
a ação DIREITA no estado 6 vale 1, a recompensa de chegar ao estado 7. Com
U(6) igual a 1, a ação DIREITA no estado 5 vale 0.9 * 1, porque a recompensa
só chega no passo seguinte e é descontada pelo gama.

A ação ESQUERDA é o contrário. Com U a 0, ESQUERDA no estado 2 vale -1, a
recompensa de chegar ao estado 1. Com U(2) igual a -1, ESQUERDA no estado 3
vale 0.9 * -1, igual a -0.9.

utilidade() repete o cálculo até a diferença entre iterações ser pequena. O
valor de cada estado desce 0.9 por cada passo de distância ao estado 7, e os
estados terminais 1 e 7 ficam a 0.

politica(U) escolhe em cada estado a ação de maior Uaccao. Nos estados de 2
a 6 é sempre DIREITA (1), porque ESQUERDA vale menos, por exemplo no estado 4
vale 0.9 * U(3) igual a 0.656 contra 0.9 * U(5) igual a 0.81.

No corredor a recompensa é +1 ao chegar ao 7 e -1 ao chegar ao 1. Os estados 1 e 7 não têm sucessores, por isso valem
0, e a recompensa conta-se na transição para lá.
A utilidade no ultimo estado é 0, pois não há razão para sair do melhor estado, ainda este não existe
"""
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.corredor.pdm_corredor import PDMCorredor


def teste_pdm():
    pdm = PDMCorredor(0.9)
    direita = AmbienteCorredor.DIREITA
    esquerda = AmbienteCorredor.ESQUERDA

    U = {}
    for s in pdm.estados:
        U[s] = 0

    print("uaccao(6, DIREITA) com U a 0 (esperado 1):", pdm.Uaccao(6, direita, U))
    print("uaccao(2, ESQUERDA) com U a 0 (esperado -1):", pdm.Uaccao(2, esquerda, U))

    U[6] = 1
    print("uaccao(5, DIREITA) com U(6) igual a 1 (esperado 0.9):", pdm.Uaccao(5, direita, U))
    U[2] = -1
    print("uaccao(3, ESQUERDA) com U(2) igual a -1 (esperado -0.9):", pdm.Uaccao(3, esquerda, U))

    U = pdm.utilidade()
    print("utilidade desce 0.9 por passo até ao estado 7 (esperado 1:0, 2:0.656, 3:0.729, 4:0.81, 5:0.9, 6:1, 7:0):",
          {s: round(U[s], 3) for s in sorted(U)})
    print("uaccao(4, ESQUERDA) e uaccao(4, DIREITA) com a utilidade final (esperado 0.656 e 0.81):",
          round(pdm.Uaccao(4, esquerda, U), 3), round(pdm.Uaccao(4, direita, U), 3))

    pi = pdm.politica(U)
    print("politica nos estados de 2 a 6 é DIREITA (esperado todos 1):",
          {s: pi[s] for s in range(2, 7)})


if __name__ == "__main__":
    teste_pdm()
"""
Testa o PDM do corredor com fator de desconto 0.9 (Processos de Decisão
Sequencial, slides 32 a 34). O modelo é conhecido, por isso tudo é calculado
sem simular o ambiente. A recompensa é 1 ao chegar ao estado 7 e -1 ao chegar
ao estado 1. Os estados 1 e 7 não têm sucessores, por isso valem 0, e a
recompensa conta-se na transição para lá.

O U é posto à mão, só para testar a fórmula T * (R + gama * U(sn)) sem
depender do resto.
    Uaccao(6, DIREITA) com U a 0 vai para 7 e vale 1 * (1 + 0.9 * 0) igual a 1.
    Uaccao(2, ESQUERDA) com U a 0 vai para 1 e vale 1 * (-1 + 0.9 * 0) igual a -1.
    Uaccao(5, DIREITA) com U(6) igual a 1 vai para 6 e vale 0 + 0.9 * 1 igual a 0.9.
    Uaccao(3, ESQUERDA) com U(2) igual a -1 vai para 2 e vale 0 + 0.9 * (-1)
    igual a -0.9.
As duas primeiras testam o termo da recompensa. As duas últimas testam o termo
descontado, em que o gama reduz o valor do que vem a seguir. Os valores de U(6)
e U(2) são artificiais, não são os reais.

O valor de cada estado é o da melhor ação, e propaga-se a partir do estado 7,
descontado 0.9 por passo.
    U(6) igual a 1, U(5) igual a 0.9 * U(6) igual a 0.9, U(4) igual a 0.81,
    U(3) igual a 0.729, U(2) igual a 0.656, e U(1) e U(7) iguais a 0.

No estado 4, com a utilidade final, DIREITA vai para 5 e vale 0.9 * 0.9 igual
a 0.81, e ESQUERDA vai para 3 e vale 0.9 * 0.729 igual a 0.656. DIREITA ganha,
e o mesmo acontece em todos os estados de 2 a 6, por isso o resultado esperado
é DIREITA (1) em todos.

Conclusões
O PDM calcula a utilidade e a política só com o modelo, por programação
dinâmica. A utilidade desce 0.9 por cada passo de distância ao estado 7, e a
política é DIREITA nos estados de 2 a 6, que é o caminho para a recompensa
positiva. Esta política serve de referência para o teste do SARSA.
"""