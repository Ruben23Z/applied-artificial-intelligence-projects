import numpy as np
from sklearn.datasets import load_digits
from funcao_ativacao.tanh import Tanh
from funcao_ativacao.sigmoid import Sigmoid
from rede_neuronal import RedeNeuronal
from utils import graficos
from utils.utilidades import treinar_epoca

"""
Teste de reconhecimento de dígitos manuscritos.

Cada imagem tem 8x8 píxeis (64 entradas, valores de 0 a 16) e a rede tem
10 saídas, uma por dígito (0-9).
É um problema de classificação geral, onde entra uma imagem, sai uma classe,
sem sequência de passos a seguir.
O alvo é bipolar (+1 na classe certa, -1 nas outras) porque a tanh dá
valores em [-1, 1] (com a sigmoide o alvo é 0/1). A classe prevista é a
saída com o valor mais alto.

No teste principal os dados são baralhados e divididos em treino (800),
validação (200) e teste (300). A validação serve para escolher a melhor
configuração; o teste só é usado uma vez, com a escolhida.


_-----------------------------------------------------------------------
Resultados do teste de reconhecimento de dígitos manuscritos

Condições: 800 exemplos de treino, 200 de validação, 300 de teste,
30 épocas, alpha = 0.01, beta = 0.5, semente 0.

Precisão (treino | validação):
    [64,16,10] tanh: 88.62% | 87.00%
    [64,32,16,10] tanh: 98.75% | 96.00%
    [64,16,10] sigmoide: 83.88% | 88.50%

Configuração escolhida: [64,32,16,10] tanh.
Precisão no teste: 95.33% (286 de 300).
Menor precisão por dígito: 9 (88.46%), 5 (88.89%), 7 (91.18%), 4 (92.59%).
Os dígitos 0, 3 e 8 foram todos classificados corretamente.

Conclusões:
- Com duas camadas escondidas a validação foi 96.00%, contra 87.00% com
uma camada escondida.
- Tanh e sigmoide diferem 1.5 pontos na validação (3 exemplos em 200),
o que não permite concluir qual é melhor dado ao valor baixo.
- A precisão no teste (95.33%) é próxima da validação (96.00%), por isso
não se observa sobreparametrização acentuada nas 30 épocas.
"""


def formatar(imagens, digitos, neg=-1):
    """Converte as imagens e os dígitos para o formato que a rede usa.
    neg: valor do alvo nas classes erradas (-1 para tanh, 0 para sigmoide)."""
    X = []
    Y = []

    for imagem in imagens:
        entrada = []
        for pixel in imagem:
            entrada.append(float(pixel) / 16)  # normaliza para [0, 1]
        X.append(entrada)

    for digito in digitos:
        vetor = []
        for i in range(10):
            if i == digito:
                vetor.append(1)
            else:
                vetor.append(neg)
        Y.append(vetor)

    return X, Y


def classe_prevista(rede, x):
    """Índice da saída mais alta, sendo a classe que a rede prevê."""
    saida = rede.propagar(x)
    return saida.index(max(saida))


def precisao(rede, X, Y):
    """Fração de exemplos em que a classe prevista coincide com a certa."""
    certos = 0
    for i in range(len(X)):
        if classe_prevista(rede, X[i]) == Y[i].index(1):
            certos += 1
    return certos / len(X)


def matriz_confusao(rede, X, Y):
    """M[i][j] = nº de exemplos da classe real i que a rede previu como j."""
    M = np.zeros((10, 10), dtype=int)
    for i in range(len(X)):
        real = Y[i].index(1)
        M[real, classe_prevista(rede, X[i])] += 1
    return M


def treinar_rede(n_treino=1797, camadas=None, func_ativacao=None, epocas=30, erro_max=0.01, alpha=0.01, beta=0.5):
    """Cria e treina uma rede com n_treino exemplos escolhidos ao acaso."""
    if camadas is None:
        camadas = [64, 16, 10]
    if func_ativacao is None:
        func_ativacao = Tanh()

    dados = load_digits()
    ordem = np.random.permutation(len(dados.data))
    imagens = dados.data[ordem][:n_treino]
    digitos = dados.target[ordem][:n_treino]

    X, Y = formatar(imagens, digitos)
    rede = RedeNeuronal(camadas, func_ativacao)
    rede.treinar(X, Y, epocas, erro_max, alpha, beta)
    return rede


def treinar_e_avaliar(forma, phi, Xt, Yt, Xv, Yv, epocas, alpha, beta):
    """Treina época a época (exemplos baralhados) e regista a precisão
    de treino e de validação em cada época."""
    rede = RedeNeuronal(forma, phi)
    treino = []
    validacao = []
    for _ in range(epocas):
        treinar_epoca(rede, Xt, Yt, alpha, beta, baralhar=True)
        treino.append(precisao(rede, Xt, Yt))
        validacao.append(precisao(rede, Xv, Yv))
    return rede, treino, validacao


if __name__ == "__main__":
    print("\n Teste dígitos manuscritos--------------")
    epocas = 30
    alpha = 0.01
    beta = 0.5

    np.random.seed(0)
    dados = load_digits()
    ordem = np.random.permutation(len(dados.data))
    imagens = dados.data[ordem]
    digitos = dados.target[ordem]
    configs = {
        "[64,16,10] tanh": ([64, 16, 10], Tanh(), -1),
        "[64,32,16,10] tanh": ([64, 32, 16, 10], Tanh(), -1),
        "[64,16,10] sigmoide": ([64, 16, 10], Sigmoid(), 0),
    }

    redes = {}
    curva_treino = {}
    curva_val = {}
    for nome in configs:
        forma, phi, neg = configs[nome]
        Xt, Yt = formatar(imagens[0:800], digitos[0:800], neg)
        Xv, Yv = formatar(imagens[800:1000], digitos[800:1000], neg)
        np.random.seed(0)
        redes[nome], curva_treino[nome], curva_val[nome] = treinar_e_avaliar(
            forma, phi, Xt, Yt, Xv, Yv, epocas, alpha, beta)
        print(f"{nome}: treino {curva_treino[nome][-1]:.2%} | validação {curva_val[nome][-1]:.2%}")

    # Escolhe a melhor configuração pela validação e só então usa o teste
    melhor = ""
    for nome in configs:
        if melhor == "" or curva_val[nome][-1] > curva_val[melhor][-1]:
            melhor = nome

    neg = configs[melhor][2]
    X_teste, Y_teste = formatar(imagens[1000:1300], digitos[1000:1300], neg)
    p_teste = precisao(redes[melhor], X_teste, Y_teste)
    print(f"\nMelhor (validação): {melhor}")
    print(f"Precisão teste: {p_teste:.2%}")

    M = matriz_confusao(redes[melhor], X_teste, Y_teste)
    for d in range(10):
        print(f"Dígito {d}: {M[d, d] / M[d].sum():.2%} ({M[d, d]}/{M[d].sum()})")

    graficos.plot_curvas(curva_val, "Precisão na validação por época", "Época", "Precisão", "digitos_validacao.png")
    graficos.plot_curvas({"treino": curva_treino[melhor], "validação": curva_val[melhor]},
                         f"Treino vs validação - {melhor}", "Época", "Precisão", "digitos_treino_vs_val.png")
    graficos.plot_matriz(M, f"Matriz de confusão (teste) - {melhor}", "digitos_confusao.png")
