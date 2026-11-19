import time
import tkinter as tk
from tkinter import ttk

from reforco.src.tetris.ambiente_tetris import AmbienteTetris
from reforco.src.tetris.graficos import GraficoLinhas
from reforco.src.tetris.treino import ALGORITMOS, Treino

CORES_CURVAS = {"Q-Learning": "#1976d2", "Dyna-Q": "#388e3c", "Q-Learning com memória": "#f57c00",
                "SARSA": "#7b1fa2", "SARSA(lambda)": "#00838f", "Dyna-Q com prioridade": "#c2185b"}
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
VELOCIDADE_MAXIMA_ANIMADA = 5  # acima disto a peça aparece logo no sítio
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
        self.estado = self.ambiente.iniciar()
        self.treinos = {}
        self.linhas_tabela = {}

        # estado do treino
        self.terminou = False
        self.perdeu = False
        self.a_treinar = False
        self.tarefa = None
        self.alvo = None  # episódio em que o treino em lote deve parar
        self.inicio = 0
        self.ultima = None  # última jogada, como (ação, recompensa)
        self.accao_pendente = None  # ação já escolhida enquanto a peça desce
        self.queda = None  # quadros da jogada que é mostrada
        self.aviso_texto = ""
        self.aviso_cor = ""
        self.aviso_ate = 0
        self.ultimo_grafico = 0

        # variáveis ligadas aos controlos
        self.nome = tk.StringVar(value="Q-Learning")
        self.velocidade = tk.IntVar(value=5)
        self.epsilon = tk.DoubleVar(value=EPSILON_INICIAL)
        self.alfa = tk.DoubleVar(value=0.1)
        self.gama = tk.DoubleVar(value=0.9)
        self.lamb = tk.DoubleVar(value=0.9)
        self.simulacoes = tk.IntVar(value=10)
        self.n_episodios = tk.IntVar(value=100)
        self.demo = tk.BooleanVar(value=False)
        self.decaimento = tk.BooleanVar(value=False)
        self.combo = tk.BooleanVar(value=False)
        self.info = tk.StringVar()
        self.pontos_texto = tk.StringVar()
        self.linhas_texto = tk.StringVar()
        self.melhor_texto = tk.StringVar()

        estilo = ttk.Style()
        estilo.configure("TButton", padding=(10, 6))
        estilo.configure("TLabelframe", padding=ESPACO)
        estilo.configure("TLabelframe.Label", font=("Arial", 11, "bold"))
        estilo.configure("TNotebook.Tab", padding=(14, 6))
        estilo.configure("Treeview", rowheight=26)

        self.construir()
        self.mudar_algoritmo()





    # ------------------------------------------------------------------
    # Janela

    def campo(self, pai, texto, linha, coluna):
        ttk.Label(pai, text=texto).grid(row=linha, column=coluna, sticky="w", padx=6, pady=6)

    def construir(self):
        # esquerd -> o jogo
        esquerda = ttk.Frame(self.raiz, padding=ESPACO)
        esquerda.grid(row=0, column=0, sticky="n")
        self.pontos = tk.Label(esquerda, textvariable=self.pontos_texto, font=("Arial", 40, "bold"), width=8)
        self.pontos.pack()
        ttk.Label(esquerda, textvariable=self.linhas_texto, font=("Arial", 12)).pack(pady=(0, 6))
        # o aviso tem altura fixa, assim não tapa o jogo nem faz a janela saltar
        self.aviso = tk.Label(esquerda, width=self.ambiente.colunas * 4, height=2,
                              bg=COR_AVISO_VAZIO, font=("Arial", 14, "bold"))
        self.aviso.pack(pady=(0, ESPACO))
        self.tabuleiro = tk.Canvas(esquerda, width=self.ambiente.colunas * TAM,
                                   height=self.ambiente.linhas * TAM, bg="white",
                                   highlightthickness=5, highlightbackground=COR_NEUTRA)
        self.tabuleiro.pack()
        ttk.Label(esquerda, text="Peça atual", font=("Arial", 11, "bold")).pack(pady=(ESPACO, 6))
        self.quadro_peca = tk.Canvas(esquerda, width=4 * TAM, height=3 * TAM, bg="white")
        self.quadro_peca.pack()

        # direita- >  controlos
        direita = ttk.Frame(self.raiz, padding=ESPACO)
        direita.grid(row=0, column=1, sticky="n")

        bloco = ttk.LabelFrame(direita, text="Algoritmo")
        bloco.pack(fill="x", pady=(0, ESPACO))
        caixa = ttk.Combobox(bloco, textvariable=self.nome, values=list(ALGORITMOS),
                             state="readonly", width=30)
        caixa.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        caixa.bind("<<ComboboxSelected>>", lambda e: self.mudar_algoritmo())
        botoes = ttk.Frame(bloco)
        botoes.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.botao = ttk.Button(botoes, text="Iniciar treino",
                                command=lambda: self.parar() if self.a_treinar else self.iniciar())
        self.botao.pack(side="left", padx=(6, 8))
        ttk.Button(botoes, text="Um passo", command=self.um_passo).pack(side="left", padx=(0, 8))
        ttk.Button(botoes, text="Reiniciar algoritmo",
                   command=lambda: self.mudar_algoritmo(True)).pack(side="left", padx=(0, 8))

        bloco = ttk.LabelFrame(direita, text="Parâmetros")
        bloco.pack(fill="x", pady=(0, ESPACO))
        self.campo(bloco, "Velocidade", 0, 0)
        tk.Scale(bloco, from_=1, to=100, orient="horizontal", variable=self.velocidade,
                 length=180).grid(row=0, column=1, columnspan=2, padx=6)
        self.campo(bloco, "Epsilon", 0, 3)
        tk.Scale(bloco, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.epsilon,
                 length=180, command=self.mudar_epsilon).grid(row=0, column=4, columnspan=2, padx=6)
        self.campo(bloco, "Alfa", 1, 0)
        ttk.Spinbox(bloco, from_=0.01, to=1, increment=0.01, format="%.2f", width=6,
                    textvariable=self.alfa, state="readonly").grid(row=1, column=1, sticky="w", padx=6)
        self.campo(bloco, "Gama", 1, 2)
        ttk.Spinbox(bloco, from_=0, to=1, increment=0.05, format="%.2f", width=6,
                    textvariable=self.gama, state="readonly").grid(row=1, column=3, sticky="w", padx=6)
        self.campo(bloco, "Simulações", 1, 4)
        ttk.Spinbox(bloco, from_=0, to=50, increment=1, width=6,
                    textvariable=self.simulacoes, state="readonly").grid(row=1, column=5, sticky="w", padx=6)
        self.campo(bloco, "Lambda", 2, 0)
        ttk.Spinbox(bloco, from_=0, to=1, increment=0.05, format="%.2f", width=6,
                    textvariable=self.lamb, state="readonly").grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(bloco, text="Alfa, gama, lambda e simulações aplicam-se ao reiniciar o algoritmo").grid(
            row=3, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 6))
        ttk.Checkbutton(bloco, text="Demonstração (sem explorar nem aprender)",
                        variable=self.demo).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(bloco, text="Epsilon decrescente",
                        variable=self.decaimento).grid(row=4, column=3, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(bloco, text="Combo (várias linhas de uma vez multiplicam os pontos)",
                        variable=self.combo, command=self.mudar_combo).grid(
            row=5, column=0, columnspan=6, sticky="w", padx=6, pady=4)

        bloco = ttk.LabelFrame(direita, text="Treino em lote")
        bloco.pack(fill="x")
        self.campo(bloco, "Episódios", 0, 0)
        ttk.Spinbox(bloco, from_=10, to=10000, increment=10, width=6,
                    textvariable=self.n_episodios, state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(bloco, text="Treinar N episódios", command=self.treinar_n).grid(row=0, column=2, padx=12)
        self.progresso = ttk.Progressbar(bloco, length=200, maximum=100)
        self.progresso.grid(row=0, column=3, padx=6)

        ttk.Label(direita, textvariable=self.info, justify="left").pack(anchor="w", pady=ESPACO)

        legenda = ttk.Frame(direita)
        legenda.pack(anchor="w", pady=(0, 6))
        for i, (nome, cor) in enumerate(CORES_CURVAS.items()):
            tk.Label(legenda, text=nome, fg=cor, font=("Arial", 10, "bold")).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 18))

        # separadores
        separadores = ttk.Notebook(direita)
        separadores.pack(fill="x")

        graficos = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(graficos, text="Gráficos")
        self.graficos = {
            "pontuacao": GraficoLinhas(graficos, "Pontuação (média móvel 20)", 320, 115),
            "recompensa": GraficoLinhas(graficos, "Recompensa por episódio (média móvel 20)", 320, 115),
            "epsilon": GraficoLinhas(graficos, "Epsilon por episódio", 320, 115),
            "estados": GraficoLinhas(graficos, "Estados conhecidos", 320, 115),
        }
        for i, grafico in enumerate(self.graficos.values()):
            grafico.grid(row=i // 2, column=i % 2, padx=3, pady=3)

        comparacao = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(comparacao, text="Comparação")
        colunas = ("Algoritmo", "Episódios", "Melhor", "Média(20)", "Estados")
        self.tabela = ttk.Treeview(comparacao, columns=colunas, show="headings", height=len(CORES_CURVAS))
        for coluna in colunas:
            self.tabela.heading(coluna, text=coluna)
            self.tabela.column(coluna, width=100, anchor="center")
        self.tabela.column("Algoritmo", width=200, anchor="w")
        self.tabela.pack(fill="x")
        ttk.Label(comparacao, textvariable=self.melhor_texto).pack(anchor="w", pady=(ESPACO, 0))

        registo = ttk.Frame(separadores, padding=ESPACO)
        separadores.add(registo, text="Registo")
        self.registo = tk.Text(registo, height=8, width=80, state="disabled", padx=8, pady=8)
        self.registo.pack()

    def log(self, texto):
        self.registo.config(state="normal")
        self.registo.insert("end", texto + "\n")
        self.registo.see("end")
        self.registo.config(state="disabled")






    # ------------------------------------------------------------------
    # Algoritmo escolhido

    def treino(self):
        """Treino do algoritmo escolhido, criado só na primeira vez que é pedido"""
        nome = self.nome.get()
        if nome not in self.treinos:
            self.treinos[nome] = Treino(nome, self.ambiente.accoes(), self.alfa.get(), self.gama.get(),
                                        self.simulacoes.get(), EPSILON_INICIAL, self.lamb.get())
        return self.treinos[nome]

    def mudar_algoritmo(self, reiniciar=False):
        """Mostra os parâmetros do algoritmo escolhido e, se pedido, recomeça-o do zero"""
        if reiniciar:
            self.treinos.pop(self.nome.get(), None)
        treino = self.treino()
        self.epsilon.set(treino.sel.epsilon)
        self.alfa.set(treino.alfa)
        self.gama.set(treino.gama)
        self.lamb.set(treino.lamb)
        self.simulacoes.set(treino.num_sim)
        self.terminou = True  # força um jogo novo no próximo passo
        self.accao_pendente = None
        self.queda = None
        self.log(("Algoritmo reiniciado " if reiniciar else "Algoritmo escolhido ") + self.nome.get())
        self.desenhar(True)

    def mudar_epsilon(self, valor):
        valor = float(valor)
        treino = self.treino()
        if abs(valor - treino.sel.epsilon) > 0.005:
            treino.sel.epsilon = valor

    def mudar_combo(self):
        self.ambiente.combo = self.combo.get()
        self.log("Combo ligado" if self.combo.get() else "Combo desligado")







    # ------------------------------------------------------------------
    # Controlo do treino

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

    def ciclo(self):
        v = self.velocidade.get()
        if v <= VELOCIDADE_MAXIMA_ANIMADA:
            # cada ciclo mostra um quadro, e no fim da queda a jogada conta
            if self.queda is None:
                treino = self.treino()
                self.recomecar_se_terminou(treino)
                # no SARSA a ação já vem escolhida do passo anterior
                if self.accao_pendente is None:
                    self.accao_pendente = self.escolher_accao(treino, self.estado)
                quadros = self.ambiente.trajectoria(self.accao_pendente)
                if quadros is not None:
                    self.queda = {"quadros": quadros, "indice": 0}
                else:
                    self.passo()  # jogada impossível, termina logo
            elif self.queda["indice"] < len(self.queda["quadros"]) - 1:
                self.queda["indice"] += 1
            else:
                self.passo()
            pausa = max(ATRASO_MINIMO_QUEDA, ATRASO_QUEDA // (v * v))
        else:
            # até à velocidade 10 encurta a pausa, depois aumentam os passos por ciclo
            for _ in range(1 if v <= 10 else (v - 10) * 10):
                self.passo()
                if self.chegou_ao_alvo():
                    break
            pausa = max(1, 1000 // (v * v))

        if self.a_treinar and self.chegou_ao_alvo():
            self.parar()
        self.desenhar()
        if self.a_treinar:
            self.tarefa = self.raiz.after(pausa, self.ciclo)






    # ------------------------------------------------------------------
    # Um passo do jogo

    def recomecar_se_terminou(self, treino):
        if self.terminou:
            self.estado = self.ambiente.iniciar()
            self.terminou = False
            self.perdeu = False
            treino.recompensa_episodio = 0
            treino.iniciar_episodio()

    def escolher_accao(self, treino, s):
        if self.demo.get():
            return treino.sel.accao_sofrega(s)  # na demonstração nunca se explora
        return treino.sel.seleccionar_accao(s)

    def passo(self):
        treino = self.treino()
        self.recomecar_se_terminou(treino)

        s = self.estado
        a = self.accao_pendente if self.accao_pendente is not None else self.escolher_accao(treino, s)
        self.accao_pendente = None
        self.queda = None

        # jogar
        linhas_antes = self.ambiente.linhas_feitas
        pontos_antes = self.ambiente.pontuacao
        sn, r, terminou = self.ambiente.executar(a)
        self.ultima = (a, r)

        feitas = self.ambiente.linhas_feitas - linhas_antes
        if feitas:
            ganhos = self.ambiente.pontuacao - pontos_antes
            if self.ambiente.multiplicador(feitas) > 1:
                self.aviso_texto, self.aviso_cor = f"COMBO x{feitas}! +{ganhos}", COR_COMBO
            else:
                self.aviso_texto, self.aviso_cor = f"LINHA! +{ganhos}", COR_LINHA
            self.aviso_ate = time.time() + SEGUNDOS_AVISO

        # aprender
        if not self.demo.get():
            if treino.on_policy:
                # a ação seguinte escolhe-se antes de aprender e é a que vai ser executada
                an = None if terminou else treino.sel.seleccionar_accao(sn)
                treino.algoritmo.aprender(s, a, r, sn, an)
                self.accao_pendente = an
            else:
                treino.algoritmo.aprender(s, a, r, sn, terminou)
            treino.passos += 1
            treino.recompensa_episodio += r

        self.estado = sn

        # fim do jogo
        if terminou:
            self.terminou = True
            self.perdeu = True
            if not self.demo.get():
                if treino.registar(self.ambiente.pontuacao):
                    self.log(f"Episódio {treino.episodios} em {treino.nome} com novo recorde de {treino.melhor}")
                if self.decaimento.get():
                    treino.sel.epsilon = max(EPSILON_MINIMO, treino.sel.epsilon * DECAIMENTO_EPSILON)
                    self.epsilon.set(treino.sel.epsilon)







    # ------------------------------------------------------------------
    # Desenho

    def desenhar_forma(self, quadro, forma, x, y, cor):
        """Desenha uma peça com o canto superior esquerdo na coluna x e na linha y"""
        for r in range(len(forma)):
            for c in range(len(forma[0])):
                if forma[r][c]:
                    quadro.create_rectangle((x + c) * TAM, (y + r) * TAM, (x + c + 1) * TAM, (y + r + 1) * TAM,
                                            fill=cor, outline="lightgray")

    def desenhar(self, forcar=False):
        ambiente = self.ambiente
        treino = self.treino()

        # aviso por cima do tabuleiro
        if self.perdeu:
            texto, cor = "GAME OVER", COR_FIM
        elif time.time() < self.aviso_ate:
            texto, cor = self.aviso_texto, self.aviso_cor
        else:
            texto, cor = "", COR_NEUTRA
        self.tabuleiro.config(highlightbackground=cor)
        self.aviso.config(text=texto, bg=cor if texto else COR_AVISO_VAZIO, fg="white")
        self.pontos.config(fg=cor if texto else "black")

        # tabuleiro, cada peça é guardada como o seu índice mais um, o zero quer dizer vazio
        self.tabuleiro.delete("all")
        for r in range(ambiente.linhas):
            for c in range(ambiente.colunas):
                valor = ambiente.grelha[r][c]
                self.desenhar_forma(self.tabuleiro, [[1]], c, r, CORES_PECAS[valor - 1] if valor else "white")
        if self.queda is not None:
            forma, x, y = self.queda["quadros"][self.queda["indice"]]
            self.desenhar_forma(self.tabuleiro, forma, x, y, CORES_PECAS[ambiente.peca])

        # peça atual
        self.quadro_peca.delete("all")
        self.desenhar_forma(self.quadro_peca, ambiente.pecas[ambiente.peca][0], 0, 0, CORES_PECAS[ambiente.peca])

        # pontuação e informação
        self.pontos_texto.set(str(ambiente.pontuacao))
        self.linhas_texto.set(f"Linhas {ambiente.linhas_feitas}     Melhor {treino.melhor}")
        if self.alvo is not None and self.a_treinar:
            self.progresso["value"] = (treino.episodios - self.inicio) / (self.alvo - self.inicio) * 100

        if self.ultima:
            a, recompensa = self.ultima
            rotacao, coluna = divmod(a, ambiente.colunas)  # o quociente é a rotação e o resto é a coluna
            ultima = f"rotação {rotacao}, coluna {coluna}, recompensa {recompensa:.1f}"
        else:
            ultima = "nenhuma"
        modo = "Demonstração" if self.demo.get() else "Treino"
        self.info.set(
            f"{modo}     Episódio {treino.episodios}     Média dos últimos 20 episódios {media_atual(treino):.1f}\n\n"
            f"Passos {treino.passos}     Epsilon {treino.sel.epsilon:.3f}\n\n"
            f"Última jogada {ultima}")

        # gráficos e tabela são mais pesados, por isso não se redesenham em todos os passos
        agora = time.time()
        if not forcar and agora - self.ultimo_grafico <= SEGUNDOS_ENTRE_GRAFICOS:
            return
        self.ultimo_grafico = agora

        for chave, campo in CAMPOS_GRAFICOS.items():
            series = {nome: (getattr(t, campo), CORES_CURVAS[nome]) for nome, t in self.treinos.items()}
            self.graficos[chave].desenhar(series)

        for nome, t in self.treinos.items():
            valores = (nome, t.episodios, t.melhor, f"{media_atual(t):.1f}", len(t.mem.obter_estados()))
            if nome in self.linhas_tabela:
                self.tabela.item(self.linhas_tabela[nome], values=valores)
            else:
                self.linhas_tabela[nome] = self.tabela.insert("", "end", values=valores)

        treinados = {nome: t for nome, t in self.treinos.items() if t.episodios > 0}
        if treinados:
            melhor = max(treinados, key=lambda nome: media_atual(treinados[nome]))
            self.melhor_texto.set(f"Melhor neste momento é o {melhor}, com média de "
                                  f"{media_atual(treinados[melhor]):.1f} nos últimos 20 episódios")


if __name__ == "__main__":
    raiz = tk.Tk()
    Interface(raiz)
    raiz.mainloop()
