"""
Gráficos simples (matplotlib) usados nos testes.
Cada função guarda uma imagem na pasta "resultados", com MOSTRAR = True mostra-a também.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "resultados"
MOSTRAR = False


def guardar(nome):
    os.makedirs(PASTA, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA, nome), dpi=150)
    if MOSTRAR:
        plt.show()
    plt.close()
    print("Gráfico guardado em", PASTA + "/" + nome)


def plot_curvas(curvas, titulo, xlabel, ylabel, nome, x=None, log=False):
    """curvas: dicionário {legenda: lista de valores}. x: eixo x (por omissão 1, 2, 3, ...)."""
    for legenda in curvas:
        valores = curvas[legenda]
        if x is None:
            plt.plot(range(1, len(valores) + 1), valores, label=legenda)
        else:
            plt.plot(x, valores, label=legenda)
    if log:
        plt.yscale("log")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    guardar(nome)


def plot_barras(rotulos, series, titulo, ylabel, nome):
    """series: dicionário {legenda: valores}, um valor por rótulo."""
    largura = 0.8 / len(series)
    posicoes = np.arange(len(rotulos))
    k = 0
    for legenda in series:
        plt.bar(posicoes + k * largura, series[legenda], largura, label=legenda)
        k += 1
    plt.xticks(posicoes + 0.4 - largura / 2, rotulos)
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    if len(series) > 1:
        plt.legend()
    guardar(nome)


def plot_matriz(M, titulo, nome):
    """Matriz de confusão: linhas = classe real, colunas = classe prevista."""
    plt.imshow(M, cmap="Blues")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, M[i, j], ha="center", va="center", fontsize=7)
    plt.xticks(range(M.shape[1]))
    plt.yticks(range(M.shape[0]))
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title(titulo)
    plt.colorbar()
    guardar(nome)


def plot_regiao(f, pontos, titulo, nome, lim=(-0.5, 1.5)):
    """Saída de f([x1, x2]) numa grelha, com os pontos (x1, x2, valor esperado) por cima."""
    xs = np.linspace(lim[0], lim[1], 100)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i, j] = f([xs[j], xs[i]])
    plt.imshow(Z, extent=[lim[0], lim[1], lim[0], lim[1]], origin="lower", cmap="coolwarm", alpha=0.4)
    plt.colorbar(label="saída")
    x1, x2, cor = [], [], []
    for p in pontos:
        x1.append(p[0])
        x2.append(p[1])
        cor.append(p[2])
    plt.scatter(x1, x2, c=cor, cmap="coolwarm", edgecolors="black", s=80)
    plt.title(titulo)
    plt.xlabel("x1")
    plt.ylabel("x2")
    guardar(nome)