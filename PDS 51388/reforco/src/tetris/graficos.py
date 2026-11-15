import tkinter as tk

MARGEM_ESQUERDA = 45
MARGEM_DIREITA = 12
MARGEM_TOPO = 32
MARGEM_BASE = 28
MAX_PONTOS = 300  # acima disto a curva fica pesada sem se notar diferença


class GraficoLinhas(tk.Canvas):
    def __init__(self, pai, titulo, largura=380, altura=170):
        super().__init__(pai, width=largura, height=altura, bg="white",
                         highlightthickness=1, highlightbackground="#cccccc")
        self.titulo = titulo
        self.largura = largura
        self.altura = altura
        self.esquerda = MARGEM_ESQUERDA
        self.direita = largura - MARGEM_DIREITA
        self.topo = MARGEM_TOPO
        self.base = altura - MARGEM_BASE

    def desenhar(self, series):
        """Series associa um nome a um par (valores, cor)"""
        self.delete("all")
        todos = [v for valores, cor in series.values() for v in valores]
        n = max([len(valores) for valores, cor in series.values()] + [2])
        minimo, maximo = self.limites(todos)

        self.desenhar_eixos(minimo, maximo, n)
        for valores, cor in series.values():
            self.desenhar_curva(valores, cor, n, minimo, maximo)

    def limites(self, valores):
        # o zero entra sempre, assim a linha de referência aparece quando há valores negativos
        minimo = min(valores + [0])
        maximo = max(valores + [0])
        if maximo - minimo < 1e-9:
            maximo = minimo + 1  # evita dividir por zero quando tudo vale o mesmo
        return minimo, maximo

    def x_do_indice(self, k, n):
        return self.esquerda + k / (n - 1) * (self.direita - self.esquerda)

    def y_do_valor(self, valor, minimo, maximo):
        fracao = (valor - minimo) / (maximo - minimo)
        return self.base - fracao * (self.base - self.topo)

    def desenhar_eixos(self, minimo, maximo, n):
        self.create_text(self.largura / 2, 14, text=self.titulo, font=("Arial", 10, "bold"))
        self.create_line(self.esquerda, self.topo, self.esquerda, self.base, self.direita, self.base)
        self.create_text(self.esquerda - 5, self.topo, text=f"{maximo:.3g}", anchor="e", font=("Arial", 8))
        self.create_text(self.esquerda - 5, self.base, text=f"{minimo:.3g}", anchor="e", font=("Arial", 8))
        self.create_text(self.direita, self.altura - 10, text=f"{n} episódios", anchor="e", font=("Arial", 8))
        if minimo < 0:
            zero = self.y_do_valor(0, minimo, maximo)
            self.create_line(self.esquerda, zero, self.direita, zero, fill="#cccccc", dash=(2, 2))

    def indices_a_desenhar(self, total):
        """Salta alguns pontos quando há muitos, mas garante sempre o último"""
        salto = max(1, total // MAX_PONTOS)
        indices = list(range(0, total, salto))
        if indices[-1] != total - 1:
            indices.append(total - 1)
        return indices

    def desenhar_curva(self, valores, cor, n, minimo, maximo):
        if len(valores) < 2:
            return
        pontos = []
        for k in self.indices_a_desenhar(len(valores)):
            pontos.append(self.x_do_indice(k, n))
            pontos.append(self.y_do_valor(valores[k], minimo, maximo))
        self.create_line(*pontos, fill=cor, width=2)