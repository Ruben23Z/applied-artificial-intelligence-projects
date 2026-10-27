"""
Funções auxiliares fora das classes da rede.
treinar_epoca e treinar_com_historico repetem o ciclo de RedeNeuronal.treinar (Parte 4 slide 43)
usando rede.adaptar, mas devolvem o erro de cada época e podem baralhar os exemplos
(SGD, Parte 4 slide 14). guardar_pesos e carregar_pesos evitam retreinar a rede.
"""
import json
import math
import numpy as np


# função em alteranativa à da desenvolvida nos slides, pois não se pretendeu altera a estrutura original,
# acabando por criar uma cópia alterada
def treinar_epoca(rede, X, Y, alpha, beta, baralhar=False):
    """Uma época de treino. Devolve o maior erro entre os exemplos."""
    indices = list(range(len(X)))
    if baralhar:
        np.random.shuffle(indices)
    erro = 0
    for i in indices:
        erro_x = rede.adaptar(X[i], Y[i], alpha, beta)
        if math.isnan(erro_x):
            erro_x = float("inf")  # max(0, nan) devolve 0 e esconderia a divergência
        erro = max(erro, erro_x)
    return erro


# função em alteranativa à da desenvolvida nos slides, pois não se pretendeu altera a estrutura original,
# acabando por criar uma cópia alterada
def treinar_com_historico(rede, X, Y, num_epocas, erro_max, alpha, beta):
    """Como RedeNeuronal.treinar, mas devolve (época em que parou, erro, erro de cada época)."""
    historico = []
    for epoca in range(1, num_epocas + 1):
        erro = treinar_epoca(rede, X, Y, alpha, beta)
        historico.append(erro)
        if erro <= erro_max:
            break
    return epoca, erro, historico


def guardar_pesos(rede, ficheiro):
    """Guarda a forma, os pesos e os pendores num ficheiro JSON."""
    pesos = []

    # A camada 0 é a camada de entrada e não tem pesos, por isso começa-se na 1
    for i in range(1, len(rede.camadas)):
        camada = rede.camadas[i]
        dados_camada = []

        for neuronio in camada.neuronios:
            lista_w = []
            for peso in neuronio.w:
                lista_w.append(float(peso))

            dados_camada.append({"w": lista_w, "b": float(neuronio.b)})

        pesos.append(dados_camada)

    dados = {"forma": rede.forma, "pesos": pesos}

    with open(ficheiro, "w") as f:
        json.dump(dados, f)


def carregar_pesos(rede, ficheiro):
    """Carrega pesos e pendores guardados; erro se a forma da rede for diferente."""
    with open(ficheiro) as f:
        dados = json.load(f)
    if dados["forma"] != rede.forma:
        raise ValueError("A forma da rede guardada é diferente")
    for c in range(1, len(rede.camadas)):
        for j in range(rede.camadas[c].ds):
            guardado = dados["pesos"][c - 1][j]
            rede.camadas[c].neuronios[j].w = guardado["w"]
            rede.camadas[c].neuronios[j].b = guardado["b"]
