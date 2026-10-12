import copy
import numpy as np
from funcao_ativacao.tanh import Tanh
from rede_neuronal import RedeNeuronal
from utils import graficos
from utils.utilidades import treinar_com_historico

"""
Exemplo dos slides 33-36 (Parte 3): reconhecimento de barras
verticais numa imagem 3x3.
Forma da rede=[9,3,1], alpha=0.2, erro_max=0.05, épocas=1000,
ativação=tanh, β=0 (sem momento, para comparação com os slides).

Referência (slide 35): erro final= 0.049937, época = 214.
Slide 36: saídas nos dados de treino e nos dados de teste (generalização).
Os valores do slide dependem da inicialização aleatória dos pesos,
por isso servem apenas de comparação aproximada.

testar_gradiente: compara o gradiente da retropropagação com o numérico.
testar_barras: treina com os dados de treino e valida com os dados de teste do slide 36.
------------------------------------------------------------------------------------------
Teste da Retropropagação
Resultados:
- Treino: parou na época 186 com erro 0.049653, nos slides foi aproximadamente na época 214,
e de erro 0.049937.
- Dados de treino: saídas entre 0.78 e 0.85 nas barras e entre -0.03 e
  0.02 nas restantes. Todas foram classificadas corretamente.
- Dados de teste do slide 36, com a média das 20 inicializações:
  y = 0.46, 0.43, 0.39, -0.02, 0.24, 0.25
  slide = 0.98, 0.96, 0.84, 0.62, -0.42, 0.27
Conclusões:
O treino chegou ao erro máximo (0.05) em número de épocas próximo do slide.
Nos dados de teste, as saídas médias diferem das do slide. A generalização depende dos pesos iniciais,
 por isso a média de várias inicializações é diferente.
"""

treino = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
]
alvos = [[0], [1], [1], [1], [0], [0], [0]]
treino_slide = [0, 0.96, 0.98, 0.95, 0, 0, 0]  # saídas da rede no slide 36

# Dados de teste do slide 36, que são imagens que a rede não viu no treino
teste = [
    [0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
]
teste_slide = [0.98, 0.96, 0.84, 0.62, -0.42, 0.27]

alpha = 0.2
erro_max = 0.05
num_epocas = 1000
beta = 0.0


def perda(rede, x, alvo):
    """L = 1/2 sum(y - alvo)**2."""
    y = rede.propagar(x)
    delta = rede.delta_saida(y, alvo)
    soma = 0
    for valor in delta:
        soma += valor ** 2
    return 0.5 * soma


def erro_relativo(a, b):
    return abs(a - b) / max(1e-8, abs(a) + abs(b))


def comparar_parametro(rede, x, alvo, objeto, nome, indice, eps):
    """Calcula o gradiente numérico de um parâmetro (peso ou pendor) do neurónio.
    Se indice for None, o parâmetro é um valor simples (pendor); caso contrário
    é a posição indice da lista de pesos."""
    if indice is None:
        original = getattr(objeto, nome)
        setattr(objeto, nome, original + eps)
        l_mais = perda(rede, x, alvo)
        setattr(objeto, nome, original - eps)
        l_menos = perda(rede, x, alvo)
        setattr(objeto, nome, original)
    else:
        lista = getattr(objeto, nome)
        original = lista[indice]
        lista[indice] = original + eps
        l_mais = perda(rede, x, alvo)
        lista[indice] = original - eps
        l_menos = perda(rede, x, alvo)
        lista[indice] = original
    return (l_mais - l_menos) / (2 * eps)


def testar_gradiente():
    """Gradiente da retropropagação vs gradiente numérico (diferenças finitas centradas)."""
    print("\n === Teste verificação do gradiente ===")
    np.random.seed(0)
    rede = RedeNeuronal([3, 4, 2], Tanh())
    x = [0.5, -0.2, 0.8]
    alvo = [1, -1]

    # Com alpha=1 e beta=0, a variação de cada parâmetro é igual a -gradiente
    copia = copy.deepcopy(rede)
    copia.adaptar(x, alvo, 1.0, 0.0)

    eps = 1e-6
    maior = 0
    for n in range(1, len(rede.camadas)):
        for j in range(rede.camadas[n].ds):
            neuronio = rede.camadas[n].neuronios[j]
            depois = copia.camadas[n].neuronios[j]

            # Pesos
            for i in range(neuronio.d):
                numerico = comparar_parametro(rede, x, alvo, neuronio, "w", i,  eps)
                analitico = -(depois.w[i] - neuronio.w[i])
                maior = max(maior, erro_relativo(analitico, numerico))

            # Pendor
            numerico = comparar_parametro(rede, x, alvo, neuronio, "b", None,  eps)
            analitico = -(depois.b - neuronio.b)
            maior = max(maior, erro_relativo(analitico, numerico))

    print(f"Maior erro do gradiente: {maior:.2e}")


def testar_barras():
    print("\n === Teste reconhecimento de barras verticais ===")

    np.random.seed(0)
    rede = RedeNeuronal([9, 3, 1], Tanh())
    epoca, erro, historico = treinar_com_historico(rede, treino, alvos, num_epocas, erro_max, alpha, beta)
    print(f"Parou na época {epoca} com erro {erro:.6f} (slide: época = 214, erro =  0.049937)")

    # Dados de treino: a rede tem de acertar todos
    print("Dados de treino:")
    for i in range(len(treino)):
        y = rede.propagar(treino[i])[0]
        print(f"  y={y:.2f} (alvo {alvos[i][0]}, slide {treino_slide[i]})")

    print("A rede aprendeu a reconhecer barras verticais corretamente")

    # Dados de teste-média de várias inicializações, porque a
    # generalização depende dos pesos iniciais
    repeticoes = 20
    soma = [0, 0, 0, 0, 0, 0]
    for semente in range(repeticoes):
        np.random.seed(semente)
        outra_rede = RedeNeuronal([9, 3, 1], Tanh())
        treinar_com_historico(outra_rede, treino, alvos, num_epocas, erro_max, alpha, beta)
        for i in range(len(teste)):
            soma[i] += outra_rede.propagar(teste[i])[0]

    media_teste = []
    for i in range(len(teste)):
        media_teste.append(soma[i] / repeticoes)

    print(f"Dados de teste (slide 36), média de {repeticoes} inicializações:")
    for i in range(len(teste)):
        print(f"  x={teste[i]} -> y={media_teste[i]:.2f} (slide {teste_slide[i]})")

    # Gráficos
    graficos.plot_curvas({"erro máximo da época": historico}, "Barras verticais: erro por época",
                         "Época", "Erro (EQM)", "barras_erro.png", log=True)
    graficos.plot_barras(["teste 1", "teste 2", "teste 3", "teste 4", "teste 5", "teste 6"],
                         {"rede (média)": media_teste, "slide 36": teste_slide},
                         "Generalização: dados de teste do slide 36", "Saída", "barras_teste.png")


if __name__ == "__main__":
    testar_gradiente()
    testar_barras()