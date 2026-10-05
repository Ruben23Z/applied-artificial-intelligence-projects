from sklearn.datasets import load_digits
from funcao_ativacao.tanh import Tanh
from rede_neuronal import RedeNeuronal

"""
Teste de reconhecimento de dígitos manuscritos

Cada imagem tem 8x8 pixeis (64 entradas, valores de 0 a 16) e a rede tem
10 saídas, uma por dígito (0-9). 
Isto é um problema de classificação geral:
entra uma imagem, sai uma classe, sem sequência de passos a seguir.
O alvo é bipolar (+1 na classe certa, -1 nas outras) porque a tanh dá
valores em [-1, 1]. A classe prevista é a saída com o valor mais alto.
"""

n_treino = 500
camadas = [64, 16, 10]
func_ativacao = Tanh()
epocas = 30
erro_max = 0.01
alpha = 0.01
beta = 0.5


def formatar(imagens, digitos):
    """Converte as imagens e os dígitos para o formato que a rede usa."""
    X = []
    Y = []
    # Para cada píxel de cada imagem normaliza os valores para [0,1]
    for imagem in imagens:
        X.append([pixel / 16 for pixel in imagem])
    # Codifica cada dígito como vetor de 10 valores: +1 na posição certa, -1 nas restantes
    for digito in digitos:
        vetor = []
        for i in range(10):
            if i == digito:
                vetor.append(1)
            else:
                vetor.append(-1)
        Y.append(vetor)
    return X, Y


def precisao(rede, X, Y):
    """Fração de exemplos em que a saída mais alta coincide com a classe certa."""
    certos = 0
    for i in range(len(X)):
        saida = rede.propagar(X[i])
        if saida.index(max(saida)) == Y[i].index(1):
            certos += 1
    return certos / len(X)


def treinar_rede(n_treino=500, camadas=None, func_ativacao=None, epocas=30, erro_max=0.01, alpha=0.01, beta=0.5):
    """Cria e treina uma rede [64, 16, 10] com os primeiros n_treino exemplos."""
    if camadas is None:
        camadas = [64, 16, 10]
    if func_ativacao is None:
        func_ativacao = Tanh()
    dados = load_digits()
    # fornece os n dados de treino e os alvos
    X, Y = formatar(dados.data[:n_treino], dados.target[:n_treino])
    rede = RedeNeuronal(camadas, func_ativacao)
    # parâmetros: épocas=30, erro_max=0.01, alpha=0.01, beta=0.5
    rede.treinar(X, Y, epocas, erro_max, alpha, beta)
    return rede


if __name__ == "__main__":
    """Treina com 500 imagens e mede a precisão em 200 imagens que a rede nunca viu."""
    print("\n === Teste dígitos manuscritos ===")
    dados = load_digits()
    rede = treinar_rede(500)

    X_treino, Y_treino = formatar(dados.data[:500], dados.target[:500])
    X_teste, Y_teste = formatar(dados.data[500:700], dados.target[500:700])

    p_teste = precisao(rede, X_teste, Y_teste)
    print(f"Precisão treino: {precisao(rede, X_treino, Y_treino):.2%}")
    print(f"Precisão teste:  {p_teste:.2%}")
    assert p_teste > 0.5, "Precisão de teste demasiado baixa"
