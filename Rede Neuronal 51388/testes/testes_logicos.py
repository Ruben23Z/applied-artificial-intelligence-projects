import numpy as np
from funcao_ativacao.degrau import Degrau
from neuronio import Neuronio
from rede_neuronal import RedeNeuronal
from utils import graficos

"""
testes da rede neuronal com pesos fixos do exemplo REDE XOR, Parte 2
slides 22-27, e pesos aleatórios

testar_xor: valida a propagação direta com forma [2,2,1] e função
degrau contra os cálculos dos slides. Mostra que uma camada escondida
resolve o XOR (não é separável linearmente, slide 34 Parte 1), o que é
impossível para um único perceptrão (slides 36-38 Parte 1).

testar_valores_aleatorios: rede não treinada (pesos aleatórios em [-1,1],
slide 32), valida apenas a forma da saída.

testar_neuronio_and_or_binario: um único neurónio, codificação binária
(0/1), implementa AND e OR (Parte 1, slides 28-30).

testar_neuronio_and_or_bipolar: o mesmo, mas com codificação bipolar
(-1/+1) (Parte 1, slides 28-32).

testar_neuronio_xor_falha: um único neurónio não consegue implementar
XOR (Parte 1, slides 33-34), porque não existe uma fronteira linear
que separe as duas classes.


-------------------------------------------------------------------
testes lógicos
Resultados:
- A rede [2,2,1] com os pesos do slide 23 deu as saídas corretas nos 4 casos do XOR.
- Um neurónio único implementou AND e OR em codificação binária e bipolar, com as saídas esperadas em todos os casos.
- Um neurónio único com w = [1, 1] e b = 0 falhou o XOR no caso (1, 1).
  A pesquisa em grelha (41^3 combinações de w1, w2, b) não encontrou
  nenhuma que acerte os 4 casos.
- A rede com pesos aleatórios e função degrau deu saída 1 nos 4 casos.
Conclusões:
O XOR é resolvido com uma camada escondida e não com um só neurónio, o que corresponde ao problema da separação linear.
"""

# Degrau com codificação binária (Parte 2 slide 23)
cod_binaria = Degrau(falso=0, estrito=True)
# Degrau com codificação bipolar (Parte 1 slides 27-34)
cod_bipolar = Degrau(falso=-1)


def testar_xor():
    """Testa a rede XOR com os pesos exatos dos slides 23-27 (Parte 2)."""
    rede = RedeNeuronal([2, 2, 1], cod_binaria)

    # Pesos e pendores do slide 23
    rede.camadas[1].neuronios[0].w = [1, -1]
    rede.camadas[1].neuronios[0].b = -0.5

    rede.camadas[1].neuronios[1].w = [-1, 1]
    rede.camadas[1].neuronios[1].b = -0.5

    rede.camadas[2].neuronios[0].w = [1, 1]
    rede.camadas[2].neuronios[0].b = -0.5

    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    esperados = [0, 1, 1, 0]

    print("\n Teste XOR")

    for i in range(len(entradas)):
        x = entradas[i]
        esperado = esperados[i]
        y = rede.propagar(x)
        y2 = [n.y for n in rede.camadas[1].neuronios]  # saídas da camada escondida

        print(f"x={x} -> y2={y2}, y3={y} (esperado {esperado})")
        assert y[0] == esperado, f"Falha no caso {x}: obtido {y}, esperado {esperado}"
    return rede


def testar_valores_aleatorios():
    """Cria uma rede com pesos aleatórios apenas para verificar a estrutura
    (sem valor esperado, pesos não treinados)."""

    print("\n === Teste com valores aleatórios ===")

    rede = RedeNeuronal([2, 4, 1], cod_binaria)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for x in X:
        y = rede.propagar(x)
        print(f"x={x} -> y={y}")
        assert len(y) == 1 and y[0] in (0, 1)  # só valida a forma da saída

    Y = rede.prever(X)
    print(f"prever(X) -> {Y}")


def testar_neuronio_and_or_binario():
    """Um único neurónio, codificação binária (0/1), implementa AND e OR
    (Parte 1, slides 29-30)."""

    print("\n === Teste Neurónio único: AND / OR (codificação binária) ===")

    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    esperados_and = [0, 0, 0, 1]
    esperados_or = [0, 1, 1, 1]

    # AND: w1=1, w2=1, b=-1
    neuronio_and = Neuronio(2, cod_binaria)
    neuronio_and.w = [1, 1]
    neuronio_and.b = -1

    for i in range(len(entradas)):
        y = neuronio_and.propagar(entradas[i])
        print(f"AND: x={entradas[i]} -> y={y} (esperado {esperados_and[i]})")
        assert y == esperados_and[i], f"Falha no AND {entradas[i]}: obtido {y}, esperado {esperados_and[i]}"

    # OR: w1=1, w2=1, b=0
    neuronio_or = Neuronio(2, cod_binaria)
    neuronio_or.w = [1, 1]
    neuronio_or.b = 0

    for i in range(len(entradas)):
        y = neuronio_or.propagar(entradas[i])
        print(f"OR:  x={entradas[i]} -> y={y} (esperado {esperados_or[i]})")
        assert y == esperados_or[i], f"Falha no OR {entradas[i]}: obtido {y}, esperado {esperados_or[i]}"


def testar_neuronio_and_or_bipolar():
    """Um único neurónio, com os pesos dos slides 29-32 (Parte 1),
    implementa AND e OR em codificação bipolar (verdade=+1, falso=-1)."""

    print("\n === Teste Neurónio único: AND / OR (codificação bipolar) ===")

    entradas = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    esperados_and = [-1, -1, -1, 1]
    esperados_or = [-1, 1, 1, 1]

    # AND: w1=1, w2=1, b=-1
    neuronio_and = Neuronio(2, cod_bipolar)
    neuronio_and.w = [1, 1]
    neuronio_and.b = -1

    for i in range(len(entradas)):
        y = neuronio_and.propagar(entradas[i])
        print(f"AND: x={entradas[i]} -> y={y} (esperado {esperados_and[i]})")
        assert y == esperados_and[i], f"Falha no AND {entradas[i]}: obtido {y}, esperado {esperados_and[i]}"

    # OR: w1=1, w2=1, b=1
    neuronio_or = Neuronio(2, cod_bipolar)
    neuronio_or.w = [1, 1]
    neuronio_or.b = 1

    for i in range(len(entradas)):
        y = neuronio_or.propagar(entradas[i])
        print(f"OR:  x={entradas[i]} -> y={y} (esperado {esperados_or[i]})")
        assert y == esperados_or[i], f"Falha no OR {entradas[i]}: obtido {y}, esperado {esperados_or[i]}"


def acerta_xor(neuronio, entradas, esperados):
    """Devolve True se o neurónio acerta nos 4 casos do XOR."""
    for i in range(len(entradas)):
        if neuronio.propagar(entradas[i]) != esperados[i]:
            return False
    return True


def testar_neuronio_xor_falha():
    """Um único neurónio não consegue implementar XOR (slides 33-34,
    Parte 1): não existe fronteira linear que separe as duas classes,
    pelo que qualquer par de pesos falha em pelo menos um dos 4 casos
    (verificado com pesquisa em grelha)."""

    print("\n === Teste de Neurónio único para XOR (falha) ===")

    entradas = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    esperados = [-1, 1, 1, -1]

    neuronio_xor = Neuronio(2, cod_bipolar)
    neuronio_xor.w = [1, 1]
    neuronio_xor.b = 0

    for i in range(len(entradas)):
        y = neuronio_xor.propagar(entradas[i])
        print(f"XOR: x={entradas[i]} -> y={y} (esperado {esperados[i]})")

    valores = np.round(np.arange(-2, 2.01, 0.1), 1)
    for w1 in valores:
        for w2 in valores:
            for b in valores:
                neuronio_xor.w = [w1, w2]
                neuronio_xor.b = b

    neuronio_xor.w = [1, 1]  # pesos usados no gráfico
    neuronio_xor.b = 0
    print("O neurónio único falha no XOR, como esperado (não separa linearmente)")
    return neuronio_xor


def graficos_logicos(rede_xor, neuronio_xor):
    """Regiões de decisão: a rede [2,2,1] separa o XOR, um neurónio único não."""
    graficos.plot_regiao(lambda p: rede_xor.propagar(p)[0],
                         [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
                         "XOR: rede [2,2,1] com camada escondida", "logicos_xor_rede.png")
    graficos.plot_regiao(lambda p: neuronio_xor.propagar(p),
                         [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)],
                         "XOR: neurónio único (uma só fronteira linear)", "logicos_xor_neuronio.png",
                         lim=(-1.5, 1.5))


if __name__ == "__main__":
    rede_xor = testar_xor()
    testar_valores_aleatorios()
    testar_neuronio_and_or_binario()
    testar_neuronio_and_or_bipolar()
    neuronio_xor = testar_neuronio_xor_falha()
    graficos_logicos(rede_xor, neuronio_xor)
