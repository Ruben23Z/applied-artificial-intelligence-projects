import time
import tkinter as tk
from tkinter import ttk

from reforco.src.tetris.ambiente_tetris import AmbienteTetris
from reforco.src.tetris.graficos import GraficoLinhas
from reforco.src.tetris.treino import ALGORITMOS, Treino

CORES_CURVAS = {"Q-Learning": "#1976d2", "Dyna-Q": "#388e3c", "Q-Learning com memória": "#f57c00"}
CORES_PECAS = ["#00bcd4", "#3f51b5", "#ff9800", "#ffeb3b", "#4caf50", "#9c27b0", "#f44336", "#e91e63"]
CAMPOS_GRAFICOS = {"pontuacao": "medias", "recompensa": "medias_recompensa",
                   "epsilon": "epsilons", "estados": "estados"}

COR_NEUTRA = "#cccccc"
COR_LINHA = "#2e7d32"
COR_COMBO = "#ef6c00"
COR_FIM = "#c62828"
COR_AVISO_VAZIO = "#eeeeee"

TAM = 32
ESPACO = 12
EPSILON_INICIAL = 0.01
EPSILON_MINIMO = 0.01
DECAIMENTO_EPSILON = 0.995
SEGUNDOS_AVISO = 0.8
SEGUNDOS_ENTRE_GRAFICOS = 0.3
VELOCIDADE_MAXIMA_ANIMADA = 5  # acima disto a peça já não se vê a mexer, aparece logo no sítio
ATRASO_QUEDA = 240  # milissegundos por movimento na velocidade 1, nas outras divide pelo quadrado
ATRASO_MINIMO_QUEDA = 10


def media_atual(treino):
    """Média móvel mais recente, ou zero se ainda não acabou nenhum episódio"""
    return treino.medias[-1] if treino.medias else 0


class Interface:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Tetris com Aprendizagem por Reforço")
        self.ambiente = AmbienteTetris()
        self.treinos = {}
        self.linhas_tabela = {}
        self.estado = self.ambiente.iniciar()
        self.criar_estado_do_treino()
        self.criar_variaveis()
        self.aplicar_estilo()
        self.construir()
        self.mudar_algoritmo()

    def criar_estado_do_treino(self):
        self.terminou = False
        self.perdeu = False
        self.a_treinar = False
        self.tarefa = None
        self.alvo = None  # episódio em que o treino em lote deve parar
        self.inicio = 0
        self.ultima = None  # última jogada feita, como (ação, recompensa)
        self.accao_pendente = None  # ação já escolhida enquanto a peça ainda desce
        self.queda = None  # jogada a ser mostrada, com todos os quadros e o índice do quadro atual
        self.aviso_texto = ""
        self.aviso_cor = ""
        self.aviso_ate = 0
        self.ultimo_grafico = 0

    def criar_variaveis(self):
        self.nome = tk.StringVar(value="Q-Learning")
        self.velocidade = tk.IntVar(value=5)
        self.epsilon = tk.DoubleVar(value=EPSILON_INICIAL)
        self.demo = tk.BooleanVar(value=False)
        self.decaimento = tk.BooleanVar(value=False)
        self.combo = tk.BooleanVar(value=False)
        self.alfa = tk.DoubleVar(value=0.1)
        self.gama = tk.DoubleVar(value=0.9)
        self.simulacoes = tk.IntVar(value=10)
        self.n_episodios = tk.IntVar(value=100)
        self.info = tk.StringVar()
        self.pontos_texto = tk.StringVar()
        self.linhas_texto = tk.StringVar()
        self.melhor_texto = tk.StringVar()

    def aplicar_estilo(self):
        estilo = ttk.Style()
        estilo.configure("TButton", padding=(10, 6))
        estilo.configure("TLabelframe", padding=ESPACO)
        estilo.configure("TLabelframe.Label", font=("Arial", 11, "bold"))
        estilo.configure("TNotebook.Tab", padding=(14, 6))
        estilo.configure("Treeview", rowheight=26)

    # ------------------------------------------------------------------
    # Construção da janela

    def construir(self):
        esquerda = ttk.Frame(self.raiz, padding=ESPACO)
        esquerda.grid(row=0, column=0, sticky="n")
        self.construir_jogo(esquerda)

        direita = ttk.Frame(self.raiz, padding=ESPACO)
        direita.grid(row=0, column=1, sticky="n")
        self.construir_controlo(direita)
        self.construir_legenda(direita)
        self.construir_separadores(direita)

    def construir_jogo(self, pai):
        self.pontos = tk.Label(pai, textvariable=self.pontos_texto, font=("Arial", 40, "bold"))
        self.pontos.pack()
        ttk.Label(pai, textvariable=self.linhas_texto, font=("Arial", 12)).pack(pady=(0, 6))
        # o aviso fica acima do tabuleiro com altura fixa, assim não tapa o jogo nem faz a janela saltar
        self.aviso = tk.Label(pai, text="", width=self.ambiente.colunas * 4, height=2,
                              bg=COR_AVISO_VAZIO, font=("Arial", 14, "bold"))
        self.aviso.pack(pady=(0, ESPACO))
        self.tabuleiro = tk.Canvas(pai, width=self.ambiente.colunas * TAM,
                                   height=self.ambiente.linhas * TAM, bg="white",
                                   highlightthickness=5, highlightbackground=COR_NEUTRA)
        self.tabuleiro.pack()
        ttk.Label(pai, text="Peça atual", font=("Arial", 11, "bold")).pack(pady=(ESPACO, 6))
        self.quadro_peca = tk.Canvas(pai, width=4 * TAM, height=3 * TAM, bg="white")
        self.quadro_peca.pack()

    def construir_controlo(self, pai):
        self.construir_bloco_algoritmo(pai)
        self.construir_bloco_parametros(pai)
        self.construir_bloco_treino(pai)
        ttk.Label(pai, textvariable=self.info, justify="left").pack(anchor="w", pady=ESPACO)

    def rotulo(self, pai, texto, linha, coluna):
        ttk.Label(pai, text=texto).grid(row=linha, column=coluna, sticky="w", padx=6, pady=6)

    def construir_bloco_algoritmo(self, pai):
        bloco = ttk.LabelFrame(pai, text="Algoritmo")
        bloco.pack(fill="x", pady=(0, ESPACO))

        caixa = ttk.Combobox(bloco, textvariable=self.nome, values=list(ALGORITMOS),
                             state="readonly", width=30)
        caixa.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        caixa.bind("<<ComboboxSelected>>", lambda e: self.escolher())

        botoes = ttk.Frame(bloco)
        botoes.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.botao = ttk.Button(botoes, text="Iniciar treino", command=self.alternar)
        self.botao.pack(side="left", padx=(6, 8))
        ttk.Button(botoes, text="Um passo", command=self.um_passo).pack(side="left", padx=(0, 8))
        ttk.Button(botoes, text="Reiniciar algoritmo", command=self.reiniciar).pack(side="left", padx=(0, 8))

    def construir_bloco_parametros(self, pai):
        bloco = ttk.LabelFrame(pai, text="Parâmetros")
        bloco.pack(fill="x", pady=(0, ESPACO))

        self.rotulo(bloco, "Velocidade", 0, 0)
        tk.Scale(bloco, from_=1, to=100, orient="horizontal", variable=self.velocidade,
                 length=180).grid(row=0, column=1, columnspan=2, padx=6)
        self.rotulo(bloco, "Epsilon", 0, 3)
        tk.Scale(bloco, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.epsilon,
                 length=180, command=self.mudar_epsilon).grid(row=0, column=4, columnspan=2, padx=6)

        self.rotulo(bloco, "Alfa", 1, 0)
        ttk.Spinbox(bloco, from_=0.01, to=1, increment=0.01, format="%.2f", width=6,
                    textvariable=self.alfa, state="readonly").grid(row=1, column=1, sticky="w", padx=6)
        self.rotulo(bloco, "Gama", 1, 2)
        ttk.Spinbox(bloco, from_=0, to=1, increment=0.05, format="%.2f", width=6,
                    textvariable=self.gama, state="readonly").grid(row=1, column=3, sticky="w", padx=6)
        self.rotulo(bloco, "Simulações", 1, 4)
        ttk.Spinbox(bloco, from_=0, to=50, increment=1, width=6,
                    textvariable=self.simulacoes, state="readonly").grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(bloco, text="Alfa, gama e simulações aplicam-se ao reiniciar o algoritmo").grid(
            row=2, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 6))

        ttk.Checkbutton(bloco, text="Demonstração (sem explorar nem aprender)",
                        variable=self.demo).grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(bloco, text="Epsilon decrescente",
                        variable=self.decaimento).grid(row=3, column=3, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(bloco, text="Combo (várias linhas de uma vez multiplicam os pontos)",
                        variable=self.combo, command=self.mudar_combo).grid(
            row=4, column=0, columnspan=6, sticky="w", padx=6, pady=4)

    def construir_bloco_treino(self, pai):
        bloco = ttk.LabelFrame(pai, text="Treino em lote")
        bloco.pack(fill="x")

        self.rotulo(bloco, "Episódios", 0, 0)
        ttk.Spinbox(bloco, from_=10, to=10000, increment=10, width=6,
                    textvariable=self.n_episodios, state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(bloco, text="Treinar N episódios", command=self.treinar_n).grid(row=0, column=2, padx=12)
        self.progresso = ttk.Progressbar(bloco, length=200, maximum=100)
        self.progresso.grid(row=0, column=3, padx=6)

    def construir_legenda(self, pai):
        legenda = ttk.Frame(pai)
        legenda.pack(anchor="w", pady=(0, 6))
        for nome, cor in CORES_CURVAS.items():
            tk.Label(legenda, text=nome, fg=cor, font=("Arial", 10, "bold")).pack(side="left", padx=(0, 18))

    def construir_separadores(self, pai):
        separadores = ttk.Notebook(pai)
        separadores.pack(fill="x")
        self.construir_graficos(separadores)
        self.construir_comparacao(separadores)
        self.construir_registo(separadores)

    def construir_graficos(self, separadores):
        separador = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(separador, text="Gráficos")
        self.graficos = {
            "pontuacao": GraficoLinhas(separador, "Pontuação (média móvel 20)"),
            "recompensa": GraficoLinhas(separador, "Recompensa por episódio (média móvel 20)"),
            "epsilon": GraficoLinhas(separador, "Epsilon por episódio"),
            "estados": GraficoLinhas(separador, "Estados conhecidos"),
        }
        for i, grafico in enumerate(self.graficos.values()):
            grafico.grid(row=i // 2, column=i % 2, padx=6, pady=6)

    def construir_comparacao(self, separadores):
        separador = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(separador, text="Comparação")
        colunas = ("Algoritmo", "Episódios", "Melhor", "Média(20)", "Estados")
        self.tabela = ttk.Treeview(separador, columns=colunas, show="headings", height=len(CORES_CURVAS))
        for coluna in colunas:
            self.tabela.heading(coluna, text=coluna)
            self.tabela.column(coluna, width=100, anchor="center")
        self.tabela.column("Algoritmo", width=200, anchor="w")
        self.tabela.pack(fill="x")
        ttk.Label(separador, textvariable=self.melhor_texto).pack(anchor="w", pady=(ESPACO, 0))

    def construir_registo(self, separadores):
        separador = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(separador, text="Registo")
        self.registo = tk.Text(separador, height=14, width=95, state="disabled", padx=8, pady=8)
        self.registo.pack()

    def log(self, texto):
        self.registo.config(state="normal")
        self.registo.insert("end", texto + "\n")
        self.registo.see("end")
        self.registo.config(state="disabled")

    # ------------------------------------------------------------------
    # Escolha e reinício do algoritmo

    def criar_treino(self, nome):
        return Treino(nome, self.ambiente.accoes(), self.alfa.get(), self.gama.get(),
                      self.simulacoes.get(), EPSILON_INICIAL)

    def treino(self):
        """Treino do algoritmo escolhido, criado só na primeira vez que é pedido"""
        nome = self.nome.get()
        if nome not in self.treinos:
            self.treinos[nome] = self.criar_treino(nome)
        return self.treinos[nome]

    def mudar_epsilon(self, valor):
        valor = float(valor)
        treino = self.treino()
        # o Scale chama isto muitas vezes, só vale a pena reagir a mudanças reais
        if abs(valor - treino.sel.epsilon) > 0.005:
            treino.sel.epsilon = valor

    def mudar_combo(self):
        self.ambiente.combo = self.combo.get()
        self.log("Combo ligado" if self.combo.get() else "Combo desligado")

    def mudar_algoritmo(self):
        treino = self.treino()
        self.epsilon.set(treino.sel.epsilon)
        self.alfa.set(treino.alfa)
        self.gama.set(treino.gama)
        self.simulacoes.set(treino.num_sim)
        self.terminou = True  # força um jogo novo no próximo passo
        self.limpar_queda()
        self.desenhar(True)

    def escolher(self):
        self.mudar_algoritmo()
        self.log("Algoritmo escolhido " + self.nome.get())

    def reiniciar(self):
        nome = self.nome.get()
        self.treinos[nome] = self.criar_treino(nome)
        self.mudar_algoritmo()
        self.log("Algoritmo reiniciado " + nome)

    # ------------------------------------------------------------------
    # Controlo do treino

    def alternar(self):
        if self.a_treinar:
            self.parar()
        else:
            self.iniciar()

    def iniciar(self, alvo=None):
        self.alvo = alvo
        self.progresso["value"] = 0
        self.a_treinar = True
        self.botao.config(text="Pausar")
        self.log("Treino iniciado " + self.nome.get())
        self.ciclo()

    def parar(self):
        self.a_treinar = False
        self.botao.config(text="Iniciar treino")
        if self.tarefa:
            self.raiz.after_cancel(self.tarefa)
            self.tarefa = None
        if self.chegou_ao_alvo():
            self.progresso["value"] = 100
        self.log(f"Treino parado no episódio {self.treino().episodios}")
        self.desenhar(True)

    def chegou_ao_alvo(self):
        return self.alvo is not None and self.treino().episodios >= self.alvo

    def treinar_n(self):
        if self.a_treinar:
            self.parar()
        self.demo.set(False)
        self.inicio = self.treino().episodios
        self.iniciar(self.inicio + self.n_episodios.get())

    def um_passo(self):
        if not self.a_treinar:
            self.passo()
            self.desenhar(True)

    def passos_por_ciclo(self, velocidade):
        # até à velocidade 10 faz-se um passo por ciclo e é a pausa que encurta
        # depois disso a pausa já é mínima, por isso aumentam os passos por ciclo
        return 1 if velocidade <= 10 else (velocidade - 10) * 10

    def pausa_entre_ciclos(self, velocidade):
        return max(1, 1000 // (velocidade * velocidade))

    def atraso_queda(self, velocidade):
        return max(ATRASO_MINIMO_QUEDA, ATRASO_QUEDA // (velocidade * velocidade))

    def ciclo(self):
        velocidade = self.velocidade.get()
        if velocidade <= VELOCIDADE_MAXIMA_ANIMADA:
            self.ciclo_animado()
            pausa = self.atraso_queda(velocidade)
        else:
            self.ciclo_rapido(velocidade)
            pausa = self.pausa_entre_ciclos(velocidade)
        self.desenhar()
        if self.a_treinar:
            self.tarefa = self.raiz.after(pausa, self.ciclo)

    def ciclo_rapido(self, velocidade):
        for _ in range(self.passos_por_ciclo(velocidade)):
            self.passo()
            if self.chegou_ao_alvo():
                self.parar()
                break

    def ciclo_animado(self):
        """Cada ciclo mostra um quadro, a peça ou desce ou faz um movimento, e no fim a jogada conta"""
        if self.queda is None:
            self.preparar_queda()
        elif self.queda["indice"] < len(self.queda["quadros"]) - 1:
            self.queda["indice"] += 1
        else:
            self.passo()
            if self.chegou_ao_alvo():
                self.parar()

    def preparar_queda(self):
        treino = self.treino()
        self.recomecar_se_terminou(treino)
        # a ação escolhe-se já, para a peça se ir mexendo enquanto cai até ao sítio onde vai pousar
        self.accao_pendente = self.escolher_accao(treino, self.estado)
        quadros = self.ambiente.trajectoria(self.accao_pendente)
        if quadros is not None:
            self.queda = {"quadros": quadros, "indice": 0}
        else:
            self.passo()  # jogada impossível, termina logo sem haver nada para animar

    # ------------------------------------------------------------------
    # Um passo do jogo

    def passo(self):
        treino = self.treino()
        self.recomecar_se_terminou(treino)

        s = self.estado
        a = self.accao_pendente if self.accao_pendente is not None else self.escolher_accao(treino, s)
        self.limpar_queda()
        sn, r, terminou = self.jogar(a)

        if not self.demo.get():
            treino.algoritmo.aprender(s, a, r, sn, terminou)
            treino.passos += 1
            treino.recompensa_episodio += r

        self.estado = sn
        if terminou:
            self.acabar_jogo(treino)

    def limpar_queda(self):
        self.accao_pendente = None
        self.queda = None

    def recomecar_se_terminou(self, treino):
        if self.terminou:
            self.estado = self.ambiente.iniciar()
            self.terminou = False
            self.perdeu = False
            treino.recompensa_episodio = 0

    def escolher_accao(self, treino, s):
        if self.demo.get():
            return treino.sel.accao_sofrega(s)  # na demonstração nunca se explora
        return treino.sel.seleccionar_accao(s)

    def jogar(self, a):
        """Executa a ação no ambiente e avisa se foram feitas linhas"""
        linhas_antes = self.ambiente.linhas_feitas
        pontos_antes = self.ambiente.pontuacao
        sn, r, terminou = self.ambiente.executar(a)
        self.ultima = (a, r)
        self.avisar_linhas(linhas_antes, pontos_antes)
        return sn, r, terminou

    def avisar_linhas(self, linhas_antes, pontos_antes):
        feitas = self.ambiente.linhas_feitas - linhas_antes
        if not feitas:
            return
        ganhos = self.ambiente.pontuacao - pontos_antes
        if self.ambiente.multiplicador(feitas) > 1:
            self.avisar(f"COMBO x{feitas}! +{ganhos}", COR_COMBO)
        else:
            self.avisar(f"LINHA! +{ganhos}", COR_LINHA)

    def avisar(self, texto, cor):
        self.aviso_texto = texto
        self.aviso_cor = cor
        self.aviso_ate = time.time() + SEGUNDOS_AVISO

    def acabar_jogo(self, treino):
        self.terminou = True
        self.perdeu = True
        if not self.demo.get():
            self.fim_episodio(treino)

    def fim_episodio(self, treino):
        if treino.registar(self.ambiente.pontuacao):
            self.log(f"Episódio {treino.episodios} em {treino.nome} com novo recorde de {treino.melhor}")
        if self.decaimento.get():
            treino.sel.epsilon = max(EPSILON_MINIMO, treino.sel.epsilon * DECAIMENTO_EPSILON)
            self.epsilon.set(treino.sel.epsilon)

    # ------------------------------------------------------------------
    # Desenho

    def desenhar(self, forcar=False):
        self.desenhar_tabuleiro()
        agora = time.time()
        # os gráficos e a tabela são mais pesados, por isso não se redesenham em todos os passos
        if forcar or agora - self.ultimo_grafico > SEGUNDOS_ENTRE_GRAFICOS:
            self.ultimo_grafico = agora
            self.desenhar_graficos()
            self.desenhar_tabela()

    def desenhar_quadrado(self, quadro, linha, coluna, cor):
        quadro.create_rectangle(coluna * TAM, linha * TAM, (coluna + 1) * TAM, (linha + 1) * TAM,
                                fill=cor, outline="lightgray")

    def desenhar_tabuleiro(self):
        texto, cor = self.aviso_atual()
        self.desenhar_celulas()
        self.atualizar_aviso(texto, cor)
        self.desenhar_peca_atual()
        self.atualizar_pontuacao(texto, cor)
        self.atualizar_progresso()
        self.atualizar_info()

    def aviso_atual(self):
        """Texto e cor do aviso por cima do tabuleiro, ou texto vazio se não houver nada a mostrar"""
        if self.perdeu:
            return "GAME OVER", COR_FIM
        if time.time() < self.aviso_ate:
            return self.aviso_texto, self.aviso_cor
        return None, COR_NEUTRA

    def desenhar_celulas(self):
        self.tabuleiro.delete("all")
        for r in range(self.ambiente.linhas):
            for c in range(self.ambiente.colunas):
                valor = self.ambiente.grelha[r][c]
                # cada peça é guardada como o seu índice mais um, o zero quer dizer vazio
                cor = CORES_PECAS[valor - 1] if valor else "white"
                self.desenhar_quadrado(self.tabuleiro, r, c, cor)
        self.desenhar_peca_a_cair()

    def desenhar_peca_a_cair(self):
        if self.queda is None:
            return
        forma, x, y = self.queda["quadros"][self.queda["indice"]]
        cor = CORES_PECAS[self.ambiente.peca]
        for r in range(len(forma)):
            for c in range(len(forma[0])):
                if forma[r][c]:
                    self.desenhar_quadrado(self.tabuleiro, y + r, x + c, cor)

    def atualizar_aviso(self, texto, cor):
        self.tabuleiro.config(highlightbackground=cor)
        if texto:
            self.aviso.config(text=texto, bg=cor, fg="white")
        else:
            self.aviso.config(text="", bg=COR_AVISO_VAZIO)

    def desenhar_peca_atual(self):
        self.quadro_peca.delete("all")
        forma = self.ambiente.pecas[self.ambiente.peca][0]
        for r in range(len(forma)):
            for c in range(len(forma[0])):
                if forma[r][c]:
                    self.desenhar_quadrado(self.quadro_peca, r, c, CORES_PECAS[self.ambiente.peca])

    def atualizar_pontuacao(self, texto, cor):
        self.pontos.config(fg=cor if texto else "black")
        self.pontos_texto.set(str(self.ambiente.pontuacao))
        self.linhas_texto.set(f"Linhas {self.ambiente.linhas_feitas}     Melhor {self.treino().melhor}")

    def atualizar_progresso(self):
        if self.alvo is not None and self.a_treinar:
            feitos = self.treino().episodios - self.inicio
            self.progresso["value"] = feitos / (self.alvo - self.inicio) * 100

    def descrever_ultima_jogada(self):
        if not self.ultima:
            return "nenhuma"
        a, recompensa = self.ultima
        # a ação é um número só, o quociente é a rotação e o resto é a coluna
        rotacao, coluna = divmod(a, self.ambiente.colunas)
        return f"rotação {rotacao}, coluna {coluna}, recompensa {recompensa:.1f}"

    def atualizar_info(self):
        treino = self.treino()
        modo = "Demonstração" if self.demo.get() else "Treino"
        self.info.set(
            f"{modo}     Episódio {treino.episodios}     Média dos últimos 20 episódios {media_atual(treino):.1f}\n\n"
            f"Passos {treino.passos}     Epsilon {treino.sel.epsilon:.3f}\n\n"
            f"Última jogada {self.descrever_ultima_jogada()}")

    def desenhar_graficos(self):
        for chave, campo in CAMPOS_GRAFICOS.items():
            series = {nome: (getattr(t, campo), CORES_CURVAS[nome]) for nome, t in self.treinos.items()}
            self.graficos[chave].desenhar(series)

    def desenhar_tabela(self):
        for nome, treino in self.treinos.items():
            self.escrever_linha_da_tabela(nome, treino)
        self.atualizar_melhor()

    def escrever_linha_da_tabela(self, nome, treino):
        valores = (nome, treino.episodios, treino.melhor, f"{media_atual(treino):.1f}",
                   len(treino.mem.obter_estados()))
        if nome in self.linhas_tabela:
            self.tabela.item(self.linhas_tabela[nome], values=valores)
        else:
            self.linhas_tabela[nome] = self.tabela.insert("", "end", values=valores)

    def atualizar_melhor(self):
        treinados = {nome: t for nome, t in self.treinos.items() if t.episodios > 0}
        if not treinados:
            return
        melhor = max(treinados, key=lambda nome: media_atual(treinados[nome]))
        self.melhor_texto.set(f"Melhor neste momento é o {melhor}, com média de "
                              f"{media_atual(treinados[melhor]):.1f} nos últimos 20 episódios")

if __name__ == "__main__":
    raiz = tk.Tk()
    Interface(raiz)
    raiz.mainloop()