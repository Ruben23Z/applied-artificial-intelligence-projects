"""
Teste do PDM do corredor com fator de desconto 0.9 (Processos de Decisão
Sequencial, slides 32 a 34). O modelo é conhecido, por isso tudo se calcula
sem simular o ambiente. A recompensa é 1 ao chegar ao estado 7 e -1 ao chegar
ao estado 1. Os estados 1 e 7 não têm sucessores nem ações, valendo 0
e a recompensa conta-se na transição para lá.

1. Uaccao (slide 33)
   Com U posto à mão, para testar T * (R + gama * U(sn)) sem depender do resto:
       Uaccao(6, DIREITA) com U a 0: 1 * (1 + 0.9 * 0) = 1
       Uaccao(2, ESQUERDA) com U a 0: 1 * (-1 + 0.9 * 0) = -1
       Uaccao(5, DIREITA) com U(6) = 1: 0 + 0.9 * 1 = 0.9
       Uaccao(3, ESQUERDA) com U(2) = -1: 0 + 0.9 * (-1) = -0.9
   As duas primeiras testam o termo da recompensa e as duas últimas o termo
   descontado. Os valores de U(6) e U(2) são artificiais.

2. utilidade (slides 32 e 33)
   O valor de cada estado é o da melhor ação e propaga-se a partir do
   estado 7, descontado 0.9 por passo:
       U(6) = 1, U(5) = 0.9, U(4) = 0.81, U(3) = 0.729, U(2) = 0.656, U(1) = U(7) = 0

3. politica (slide 34)
   No estado 4, DIREITA vale 0.9 * 0.9 = 0.81 e ESQUERDA vale 0.9 * 0.729 = 0.656.
   O mesmo acontece nos estados de 2 a 6, por isso a política esperada é
   DIREITA (1) nesses estados. Os terminais não aparecem, porque não têm ações.
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

    print("politica (esperado só os estados 2 a 6, todos DIREITA = 1):", pi)


if __name__ == "__main__":
    teste_pdm()
"""
Resultados e conclusões

uaccao(6, DIREITA) com U a 0 (esperado 1): 1
uaccao(2, ESQUERDA) com U a 0 (esperado -1): -1
uaccao(5, DIREITA) com U(6) igual a 1 (esperado 0.9): 0.9
uaccao(3, ESQUERDA) com U(2) igual a -1 (esperado -0.9): -0.9
utilidade desce 0.9 por passo até ao estado 7 (...): {1: 0, 2: 0.656, 3: 0.729, 4: 0.81, 5: 0.9, 6: 1.0, 7: 0}
uaccao(4, ESQUERDA) e uaccao(4, DIREITA) com a utilidade final (esperado 0.656 e 0.81): 0.656 0.81
politica (esperado só os estados 2 a 6, todos DIREITA = 1): {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

O PDM calcula a utilidade e a política só com o modelo, por programação
dinâmica. A utilidade desce 0.9 por cada passo de distância ao estado 7 e a
política é DIREITA nos estados de 2 a 6. Os estados terminais não aparecem na
política. Esta política serve de referência para o teste do SARSA.
"""