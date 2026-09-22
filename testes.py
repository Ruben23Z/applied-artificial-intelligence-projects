from rede_neuronal import RedeNeuronal


def degrau(x):
    return 1 if x > 0 else 0


def testar_xor():
    """Testa a rede XOR com os pesos exatos do slide 23-27 (aula 1)."""
    rede = RedeNeuronal([2, 2, 1], degrau)

    # Pesos e pendores do slide 23
    rede.camadas[1].neuronios[0].w = [1, -1]  # camada 2, neurónio 1 (y1): w11=1, w21=-1
    rede.camadas[1].neuronios[0].b = -0.5  # camada 2, neurónio 1: pendor b1

    rede.camadas[1].neuronios[1].w = [-1, 1]  # camada 2, neurónio 2 (y2): w12=-1, w22=1
    rede.camadas[1].neuronios[1].b = -0.5  # camada 2, neurónio 2: pendor b2

    rede.camadas[2].neuronios[0].w = [1, 1]  # camada 3, neurónio único (y3): w11=1, w21=1
    rede.camadas[2].neuronios[0].b = -0.5  # camada 3, neurónio único: pendor b3

    function_xor = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    print("\n === Teste XOR ===")

    for x, esperado in function_xor:  # percorre os 4 casos (entrada, saída esperada) do XOR
        y = rede.propagar(x)  # propaga x por todas as camadas. Com y = saída final da rede
        y2 = [n.y for n in rede.camadas[1].neuronios]  # saída de cada neurónio da camada escondida(valor intermédio)

        ok = "OK" if y[0] == esperado else "FALHOU"  # compara saída obtida com a esperada, só para o print

        print(
            f"x={x} -> y2={y2}, y3={y} (esperado {esperado}) [{ok}]")  # mostra entrada, saída escondida, saída final e resultado
        assert y[0] == esperado, f"Falha no caso {x}: obtido {y}, esperado {esperado}"  # valida o teste; para o programa se falhar


def testar_valores_aleatorios():
    """Cria redes com pesos aleatórios apenas para verificar integridade
    estrutural (sem valor esperado, pesos não treinados)."""

    print("\n === Teste com valores aleatórios ===")

    rede = RedeNeuronal([2, 4, 1], degrau)  # rede não treinada: 2 entradas, 4 escondidos, 1 saída, pesos aleatórios

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Conjunto de entradas em lote, devido á camada de entrada de tamanho 2

    for x in X:  # testa as 4 combinações de entrada uma a uma
        y = rede.propagar(x)  # propaga x pela rede
        print(f"x={x} -> y={y}")
        assert len(y) == 1 and y[0] in (0, 1)  # valida só a forma: 1 saída, valor booleano (não valida se está "certo")

    Y = rede.prever(X)  # prever() propaga cada x de X e devolve a lista de saídas correspondente
    print(f"prever(X) -> {Y}")
    assert len(Y) == 4  # confirma que devolveu uma saída por cada entrada



if __name__ == "__main__":
    # testar_xor()
    testar_valores_aleatorios()

