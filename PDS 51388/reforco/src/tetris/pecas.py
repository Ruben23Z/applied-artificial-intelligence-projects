

#Formas das peças em array
FORMAS = [
    [[1, 1, 1, 1]],
    [[0, 1], [0, 1], [1, 1]],
    [[1, 0], [1, 0], [1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 0], [1, 1]],
]


def rodar(forma):
    """Roda a peça 90 graus no sentido dos ponteiros do relógio"""
    linhas_originais = len(forma)
    colunas_originais = len(forma[0])
    nova = []
    for y in range(colunas_originais):
        linha = []
        for x in range(linhas_originais):
            # cada coluna da peça antiga passa a ser uma linha da nova, lida de baixo para cima
            linha.append(forma[linhas_originais - x - 1][y])
        nova.append(linha)
    return nova


def rotacoes(forma):
    """Devolve as quatro rotações da peça, começando na posição original"""
    formas = [forma]
    for _ in range(3):
        formas.append(rodar(formas[-1]))
    return formas