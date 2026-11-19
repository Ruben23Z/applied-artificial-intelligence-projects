import random
from collections import deque

from reforco.src.tetris.pecas import FORMAS, rotacoes

PONTOS_POR_LINHA = 100
RECOMPENSA_POR_LINHA = 10
PENALIZACAO_POR_BURACO = 1
PENALIZACAO_POR_ALTURA = 0.5
PENALIZACAO_FIM = 10


class AmbienteTetris:
    """
    Uma posição da peça é um tuplo (rotação, coluna, linha)
    A peça nasce ao centro da linha de cima e só pode descer, andar para os lados ou rodar,
    sempre sem tocar em nada
    Para cada ação do agente escolhe a rotação e a coluna onde a peça deve pousar
    """

    def __init__(self, colunas=6, linhas=12, combo=False):
        self.colunas = colunas
        self.linhas = linhas
        self.combo = combo  # com combo ligado, várias linhas de uma vez valem mais pontos
        self.pecas = [rotacoes(f) for f in FORMAS]
        self.grelha = []
        self.peca = 0
        self.pontuacao = 0
        self.linhas_feitas = 0
        self.iniciar()

    def accoes(self):
        # cada ação é um número só, que junta a rotação e a coluna onde a peça deve pousar
        return list(range(4 * self.colunas))

    def iniciar(self):
        self.grelha = [[0] * self.colunas for _ in range(self.linhas)]
        self.pontuacao = 0
        self.linhas_feitas = 0
        self.peca = random.randrange(len(self.pecas))
        return self.estado()

    def alturas(self):
        """Altura de cada coluna, medida desde o chão até à primeira célula ocupada"""
        alturas = []
        for c in range(self.colunas):
            altura = 0
            for r in range(self.linhas):
                if self.grelha[r][c]:
                    altura = self.linhas - r
                    break
            alturas.append(altura)
        return alturas

    def estado(self):
        """
        O estado é a peça atual e o desnível entre as colunas vizinhas
        Cada desnível é limitado a -1, 0 ou 1 para o número de estados não crescer demais
        """
        alturas = self.alturas()
        desniveis = []
        for c in range(self.colunas - 1):
            desnivel = alturas[c + 1] - alturas[c]
            desniveis.append(max(-1, min(1, desnivel)))
        return (self.peca, tuple(desniveis))

    def buracos(self):
        """Conta as células vazias que têm uma célula ocupada por cima"""
        total = 0
        for c in range(self.colunas):
            encontrou_peca = False
            for r in range(self.linhas):
                if self.grelha[r][c]:
                    encontrou_peca = True
                elif encontrou_peca:
                    total += 1
        return total

    def cabe(self, forma, x, y):
        for r in range(len(forma)):
            for c in range(len(forma[0])):
                if forma[r][c]:
                    if y + r >= self.linhas or self.grelha[y + r][x + c]:
                        return False
        return True

    def limpar_linhas(self):
        """Remove as linhas cheias e põe linhas vazias no topo para a grelha manter o tamanho"""
        restantes = [linha for linha in self.grelha if not all(linha)]
        limpas = self.linhas - len(restantes)
        vazias = [[0] * self.colunas for _ in range(limpas)]
        self.grelha = vazias + restantes
        return limpas

    def multiplicador(self, limpas):
        """Com combo, limpar duas ou mais linhas de uma vez multiplica os pontos pelo número de linhas"""
        if self.combo and limpas > 1:
            return limpas
        return 1

    def pontos_da_jogada(self, limpas):
        return PONTOS_POR_LINHA * limpas * self.multiplicador(limpas)







    # ------------------------------------------------------------------
    # Movimentos da peça

    def forma_em(self, rotacao):
        return self.pecas[self.peca][rotacao % 4]

    def posicao_inicial(self):
        """A peça nasce ao centro, na linha de cima e sem estar rodada"""
        forma = self.forma_em(0)
        return 0, (self.colunas - len(forma[0])) // 2, 0

    def nasce_livre(self):
        rotacao, x, y = self.posicao_inicial()
        return self.cabe(self.forma_em(rotacao), x, y)

    def rodada(self, posicao):
        """Roda a peça 90 graus no sítio onde está, ou devolve None se bater em alguma coisa"""
        rotacao, x, y = posicao
        seguinte = (rotacao + 1) % 4
        forma = self.forma_em(seguinte)
        # a peça rodada pode ficar mais larga, nesse caso se encosta à parede em vez de a atravessar
        novo_x = min(x, self.colunas - len(forma[0]))
        if self.cabe(forma, novo_x, y):
            return seguinte, novo_x, y
        return None

    def movimentos(self, posicao):
        """Tudo o que a peça pode fazer a partir de uma posição, cada movimento com o nome"""
        rotacao, x, y = posicao
        forma = self.forma_em(rotacao)
        possiveis = []
        if self.cabe(forma, x, y + 1):
            possiveis.append(("descer", (rotacao, x, y + 1)))
        if x > 0 and self.cabe(forma, x - 1, y):
            possiveis.append(("esquerda", (rotacao, x - 1, y)))
        if x + len(forma[0]) < self.colunas and self.cabe(forma, x + 1, y):
            possiveis.append(("direita", (rotacao, x + 1, y)))
        rodada = self.rodada(posicao)
        if rodada is not None:
            possiveis.append(("rodar", rodada))
        return possiveis

    def linha_de_queda(self, forma, x):
        """Desce a peça até já não caber mais abaixo e devolve a linha onde ela para"""
        y = 0
        while self.cabe(forma, x, y + 1):
            y += 1
        return y

    def alvo(self, a):
        """Onde o agente quer que a peça pouse, ou None se nem sequer cabe no topo dessa coluna"""
        rotacao, coluna = divmod(a, self.colunas)
        forma = self.forma_em(rotacao)
        # se a peça sair da grelha pela direita, encosta-se ao lado direito
        x = min(coluna, self.colunas - len(forma[0]))
        if not self.cabe(forma, x, 0):
            return None
        return rotacao, forma, x, self.linha_de_queda(forma, x)

    def caminho_direto(self, rotacao, x_alvo):
        """
        Verifica o caminho mais simples, rodar e andar na linha de cima e depois deixar cair
        É o que quase sempre resulta, sendo testado primeiro por ser muito mais barato
        """
        posicao = self.posicao_inicial()
        for _ in range(rotacao):
            posicao = self.rodada(posicao)
            if posicao is None:
                return False

        rotacao_final, x, _ = posicao
        forma = self.forma_em(rotacao_final)
        sentido = 1 if x_alvo > x else -1
        while x != x_alvo:
            x += sentido
            if not self.cabe(forma, x, 0):
                return False
        return True

    def explorar(self):
        """
        Pesquisa em largura de todas as posições a que a peça consegue chegar
        Guarda a distância de cada uma e, para cada uma, as posições que a podem ter precedido
        num caminho mais curto
        """
        inicio = self.posicao_inicial()
        distancias = {inicio: 0}
        antes = {inicio: []}
        fila = deque([inicio])
        while fila:
            atual = fila.popleft()
            for movimento, seguinte in self.movimentos(atual):
                if seguinte not in distancias:
                    distancias[seguinte] = distancias[atual] + 1
                    antes[seguinte] = []
                    fila.append(seguinte)
                if distancias[seguinte] == distancias[atual] + 1:
                    antes[seguinte].append((atual, movimento))
        return distancias, antes

    def posicoes_com_forma(self, distancias, forma, x, y):
        """Posições alcançáveis com esta forma, coluna e linha, mesmo que a rotação seja outra"""
        resultado = []

        for p in distancias:
            if self.forma_em(p[0]) == forma and p[1] == x and p[2] == y:
                resultado.append(p)

        return resultado
    def alcanca(self, rotacao, forma, x, y):
        if self.caminho_direto(rotacao, x):
            return True
        distancias, _ = self.explorar()
        return bool(self.posicoes_com_forma(distancias, forma, x, y))

    def destino(self, a):
        """
        Devolve a forma, a coluna e a linha onde a peça vai mesmo pousar com a ação a
        Se não houver caminho livre até ao alvo, a peça cai a direito onde nasceu
        """
        alvo = self.alvo(a)
        if alvo is not None and self.alcanca(*alvo):
            rotacao, forma, x, y = alvo
            return forma, x, y

        _, x, _ = self.posicao_inicial()
        forma = self.forma_em(0)
        return forma, x, self.linha_de_queda(forma, x)

    def trajectoria(self, a):
        """
        Devolve os quadros (forma, coluna, linha) que a peça percorre desde que nasce até pousar
        Serve só para mostrar a jogada, quem decide o resultado é o executar
        """
        if not self.nasce_livre():
            return None
        forma, x, y = self.destino(a)
        inicio = self.posicao_inicial()
        anterior = {inicio: None}
        fila = deque([inicio])

        while fila:
            atual = fila.popleft()
            if self.forma_em(atual[0]) == forma and atual[1] == x and atual[2] == y:
                break
            for _, seguinte in self.movimentos(atual):
                if seguinte not in anterior:
                    anterior[seguinte] = atual
                    fila.append(seguinte)
        caminho = []
        while atual is not None:
            caminho.append((self.forma_em(atual[0]), atual[1], atual[2]))
            atual = anterior[atual]
        return caminho[::-1]

    def caminho_alternado(self, objetivo, distancias, antes):
        """
        Escolhe, entre os caminhos mais curtos, um que alterna entre descer e fazer um movimento
        Assim os movimentos vão a acontecer enquanto a peça cai, e não todos no topo
        """
        # posições que fazem parte de algum caminho mais curto até ao objetivo
        boas = {objetivo}
        por_visitar = [objetivo]
        while por_visitar:
            posicao = por_visitar.pop()
            for anterior, movimento in antes[posicao]:
                if anterior not in boas:
                    boas.add(anterior)
                    por_visitar.append(anterior)

        atual = self.posicao_inicial()
        caminho = [atual]
        ultimo = None
        while atual != objetivo:
            # movimentos que continuam num caminho mais curto
            opcoes = []
            for movimento, seguinte in self.movimentos(atual):
                if seguinte in boas and distancias[seguinte] == distancias[atual] + 1:
                    opcoes.append((seguinte, movimento))

            # após descer prefere-se outro movimento, e depois de outro movimento prefere-se descer
            preferidas = []
            for seguinte, movimento in opcoes:
                if ultimo == "descer" and movimento != "descer":
                    preferidas.append((seguinte, movimento))
                elif ultimo != "descer" and movimento == "descer":
                    preferidas.append((seguinte, movimento))

            if preferidas:
                atual, ultimo = preferidas[0]
            else:
                atual, ultimo = opcoes[0]
            caminho.append(atual)
        return caminho






    # ------------------------------------------------------------------
    # Jogada

    def pousar(self, forma, x, y):
        for r in range(len(forma)):
            for c in range(len(forma[0])):
                if forma[r][c]:
                    self.grelha[y + r][x + c] = self.peca + 1

    def calcular_recompensa(self, limpas, buracos_antes, altura_antes):
        buracos_novos = self.buracos() - buracos_antes
        altura_nova = max(self.alturas()) - altura_antes
        return (RECOMPENSA_POR_LINHA * limpas
                - PENALIZACAO_POR_BURACO * buracos_novos
                - PENALIZACAO_POR_ALTURA * altura_nova)

    def executar(self, a):
        # se a peça já não tem espaço para nascer, o jogo acabou
        if not self.nasce_livre():
            return self.estado(), -PENALIZACAO_FIM, True

        forma, x, y = self.destino(a)
        buracos_antes = self.buracos()
        altura_antes = max(self.alturas())

        self.pousar(forma, x, y)
        limpas = self.limpar_linhas()
        self.pontuacao += self.pontos_da_jogada(limpas)
        self.linhas_feitas += limpas

        recompensa = self.calcular_recompensa(limpas, buracos_antes, altura_antes)
        terminou = any(self.grelha[0])
        if terminou:
            recompensa -= PENALIZACAO_FIM

        self.peca = random.randrange(len(self.pecas))
        return self.estado(), recompensa, terminou
