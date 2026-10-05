from funcao_ativacao.func_ativacao import FuncAtivacao
from neuronio import Neuronio
from rede_neuronal import RedeNeuronal

"""
Testes da rede neuronal (pesos fixos do exemplo REDE XOR, Parte 2
slides 22-27, e pesos aleatórios).

testar_xor: valida a propagação direta com forma [2,2,1] e função
degrau contra os cálculos dos slides. Demonstra que uma camada
escondida resolve o XOR (não separável linearmente, slide 34 Parte 1),
algo impossível a um único perceptrão (slides 36-38 Parte 1).

testar_valores_aleatorios: rede não treinada (pesos random [-1,1],
slide 32), valida só a forma da saída.

testar_neuronio_and_or_binario: um único neurónio, codificação binária
(0/1), implementa AND e OR (Parte 1, slides 28-30).

testar_neuronio_and_or_bipolar: o mesmo, com codificação bipolar
(-1/+1) (Parte 1, slides 28-32).

testar_neuronio_xor_falha: o mesmo tipo de neurónio único não consegue
implementar XOR (Parte 1, slides 33-34) — não existe fronteira linear
única que separe as duas classes.
"""


class CodBinaria(FuncAtivacao):
    """Degrau com codificação binária (Parte 2 slide 23, verdade=1, falso=0):
    f(h)=1 se h>0, 0 se h<=0. A definição geral (Parte 1 slide 23) usa h>=0,
    mas aqui replica-se a condição exata do exemplo para dar os valores dos slides."""

    def calcular(self, h):
        return 1 if h > 0 else 0

    def derivada(self, h):
        return 0  # o degrau não é diferenciável (Parte 1 slide 24); aqui só se testa a propagação


class CodBipolar(FuncAtivacao):
    """Degrau com codificação bipolar (Parte 1 slides 27-34, verdade=+1, falso=-1):
    y=+1 se h>=0, -1 caso contrário."""

    def calcular(self, h):
        return 1 if h >= 0 else -1

    def derivada(self, h):
        return 0


# Instâncias prontas a passar como φ, assim o resto dos Testes fica igual
cod_binaria = CodBinaria()
cod_bipolar = CodBipolar()


def testar_xor():
    """Testa a rede XOR com os pesos exatos do slides 23-27 (Parte 2)."""
    rede = RedeNeuronal([2, 2, 1], cod_binaria)

    # Pesos e pendores do slide 23
    rede.camadas[1].neuronios[0].w = [1, -1]  # camada 2, neurónio 1 (y1): w11=1, w21=-1
    rede.camadas[1].neuronios[0].b = -0.5  # camada 2, neurónio 1: pendor b1

    rede.camadas[1].neuronios[1].w = [-1, 1]  # camada 2, neurónio 2 (y2): w12=-1, w22=1
    rede.camadas[1].neuronios[1].b = -0.5  # camada 2, neurónio 2: pendor b2

    rede.camadas[2].neuronios[0].w = [1, 1]  # camada 3, neurónio único (y3): w11=1, w21=1
    rede.camadas[2].neuronios[0].b = -0.5  # camada 3, neurónio único: pendor b3

    xor = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    print("\n === Teste XOR ===")

    for x, esperado in xor:  # percorre os 4 casos (entrada, saída esperada) do XOR
        y = rede.propagar(x)  # propaga x por todas as camadas. Com y = saída final da rede
        y2 = [n.y for n in rede.camadas[1].neuronios]  # saída de cada neurónio da camada escondida(valor intermédio)

        print(
            f"x={x} -> y2={y2}, y3={y} (esperado {esperado})")  # mostra entrada, saída escondida, saída final e resultado
        assert y[
                   0] == esperado, f"Falha no caso {x}: obtido {y}, esperado {esperado}"  # valida o teste; para o programa se falhar


def testar_valores_aleatorios():
    """Cria redes com pesos aleatórios apenas para verificar integridade
    estrutural (sem valor esperado, pesos não treinados)."""

    print("\n === Teste com valores aleatórios ===")

    rede = RedeNeuronal([2, 4, 1],
                        cod_binaria)  # rede não treinada: 2 entradas, 4 escondidos, 1 saída, pesos aleatórios

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Conjunto de entradas em lote, devido á camada de entrada de tamanho 2

    for x in X:  # testa as 4 combinações de entrada uma a uma
        y = rede.propagar(x)  # propaga x pela rede
        print(f"x={x} -> y={y}")
        assert len(y) == 1 and y[0] in (0, 1)  # valida só a forma: 1 saída, valor booleano (não valida se está "certo")

    Y = rede.prever(X)  # prever() propaga cada x de X e devolve a lista de saídas correspondente
    print(f"prever(X) -> {Y}")
    assert len(Y) == 4  # confirma que devolveu uma saída por cada entrada


def testar_neuronio_and_or_binario():
    """Um único neurónio, codificação binária (0/1), implementa AND e OR
    (Parte 1, slide 29: tabela A B A.B / A+B; slide 30: w1=1,w2=1,b=-1
    para AND, threshold em h>0)."""

    print("\n === Teste Neurónio único: AND / OR (codificação binária) ===")

    # AND (slide 29-30): w1=1, w2=1, b=-1
    and_casos = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ]
    neuronio_and = Neuronio(2, cod_binaria)
    neuronio_and.w = [1, 1]
    neuronio_and.b = -1

    for a, b, esperado in and_casos:
        y = neuronio_and.propagar([a, b])
        print(f"AND: x=({a},{b}) -> y={y} (esperado {esperado})")
        assert y == esperado, f"Falha no AND ({a},{b}): obtido {y}, esperado {esperado}"

    # OR (slide 29-31): w1=1, w2=1, b=0
    or_casos = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    neuronio_or = Neuronio(2, cod_binaria)
    neuronio_or.w = [1, 1]
    neuronio_or.b = 0

    for a, b, esperado in or_casos:
        y = neuronio_or.propagar([a, b])
        print(f"OR:  x=({a},{b}) -> y={y} (esperado {esperado})")
        assert y == esperado, f"Falha no OR ({a},{b}): obtido {y}, esperado {esperado}"


def testar_neuronio_and_or_bipolar():
    """Um único neurónio, com os pesos exatos dos slides 29-32 (Parte 1),
    implementa AND e OR em codificação bipolar (verdade=+1, falso=-1)."""

    print("\n === Teste Neurónio único: AND / OR (codificação bipolar) ===")

    # AND (slide 29-30): w1=1, w2=1, b=-1
    and_casos = [
        (-1, -1, -1),
        (-1, 1, -1),
        (1, -1, -1),
        (1, 1, 1),
    ]
    neuronio_and = Neuronio(2, cod_bipolar)
    neuronio_and.w = [1, 1]
    neuronio_and.b = -1

    for a, b, esperado in and_casos:
        y = neuronio_and.propagar([a, b])
        print(f"AND: x=({a},{b}) -> y={y} (esperado {esperado})")
        assert y == esperado, f"Falha no AND ({a},{b}): obtido {y}, esperado {esperado}"

    # OR (slide 31-32): w1=1, w2=1, b=1
    or_casos = [
        (-1, -1, -1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, 1),
    ]
    neuronio_or = Neuronio(2, cod_bipolar)
    neuronio_or.w = [1, 1]
    neuronio_or.b = 1

    for a, b, esperado in or_casos:
        y = neuronio_or.propagar([a, b])
        print(f"OR:  x=({a},{b}) -> y={y} (esperado {esperado})")
        assert y == esperado, f"Falha no OR ({a},{b}): obtido {y}, esperado {esperado}"


def testar_neuronio_xor_falha():
    """Um único neurónio não consegue implementar XOR (slides 33-34,
    Parte 1): não existe fronteira linear que separe as duas classes,
    pelo que qualquer par de pesos falha em pelo menos um dos 4 casos."""

    print("\n === Teste de Neurónio único para XOR(falha) ===")

    xor_casos = [
        (-1, -1, -1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    ]

    neuronio_xor = Neuronio(2, cod_bipolar)
    neuronio_xor.w = [1, 1]
    neuronio_xor.b = 0

    algum_erro = False
    for a, b, esperado in xor_casos:
        y = neuronio_xor.propagar([a, b])
        print(f"XOR: x=({a},{b}) -> y={y} (esperado {esperado})")
        if y != esperado:
            algum_erro = True

    assert algum_erro, (
        "Inesperado: um único neurónio não deveria conseguir resolver XOR "
        "(problema da separabilidade linear, slide 34 Parte 1)"
    )
    print("O neurónio único falha no XOR, como esperado (não separa linearmente)")


if __name__ == "__main__":
    testar_xor()
    testar_valores_aleatorios()
    testar_neuronio_and_or_binario()
    testar_neuronio_and_or_bipolar()
    testar_neuronio_xor_falha()