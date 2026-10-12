import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from testes.testes_digitos import treinar_rede
from funcao_ativacao.tanh import Tanh
from rede_neuronal import RedeNeuronal
from utils.utilidades import carregar_pesos, guardar_pesos

"""
Desenha um dígito ao arrastar o rato e a rede classifica.
O desenho é feito em alta resolução e só depois reduzido para 8x8
(recorte, centragem e média por blocos), o que o aproxima das imagens de treino.
A janela abre primeiro; a rede é carregada de ficheiro ou treinada depois (e guardada).
Se mudares os parâmetros de treino, apaga o ficheiro de pesos para a rede ser retreinada.
"""

camadas = [64, 32, 16, 10]
func_ativacao = Tanh()
epocas = 30
erro_max = 0.01
alpha = 0.01
beta = 0.5
n_treino = 1797  # o dataset tem 1797 imagens

TAM = 720  # lado do canvas em píxeis
PINCEL = 25  # espessura do traço
FICHEIRO = "pesos_digitos.json"


def obter_rede():
    """Carrega os pesos guardados; se não existirem (ou a forma mudou), treina e guarda."""
    rede = RedeNeuronal(camadas, func_ativacao)
    try:
        carregar_pesos(rede, FICHEIRO)
    except (FileNotFoundError, ValueError):
        rede = treinar_rede(n_treino, camadas, func_ativacao, epocas, erro_max, alpha, beta)
        guardar_pesos(rede, FICHEIRO)
    return rede


def imagem_para_entrada(img):
    """
    Converte uma imagem em tons de cinzento (traço claro sobre fundo preto)
    num vetor de 64 valores em [0, 1], na mesma escala do treino (pixel/16).
    O dataset foi gerado a partir de bitmaps binários 32x32, em que cada
    píxel 8x8 é a contagem de píxeis ligados num bloco 4x4 (0 a 16).
    Aqui: recorta, centra, adiciona margem, binariza a 32x32 e faz a
    média por blocos de 4x4.
    """
    caixa = img.getbbox()
    if caixa is None:
        return None
    img = img.crop(caixa)

    lado = max(img.size)
    quadrado = Image.new("L", (lado, lado), 0)
    quadrado.paste(img, ((lado - img.width) // 2, (lado - img.height) // 2))

    quadrado = ImageOps.expand(quadrado, border=lado // 8, fill=0)

    # Bitmap binário 32x32, como o original do dataset
    bitmap = quadrado.resize((32, 32), Image.BOX)
    bitmap = bitmap.point(lambda p: 255 if p >= 128 else 0)

    # Média por blocos 4x4 = (nº de píxeis ligados) / 16, igual a pixel/16 do treino
    pequena = bitmap.resize((8, 8), Image.BOX)
    entrada = [p / 255 for p in pequena.getdata()]

    print("\n ---------------")
    print(np.array(entrada).reshape(8, 8).round(1))
    return entrada



class AppDigitos:

    def __init__(self):
        self.imagem = Image.new("L", (TAM, TAM), 0)
        self.desenho = ImageDraw.Draw(self.imagem)
        self.ultimo = None
        self.rede = None

        self.janela = tk.Tk()
        self.janela.title("Dígitos")
        self.canvas = tk.Canvas(self.janela, width=TAM, height=TAM, bg="white")
        self.canvas.pack()
        self.resultado = tk.Label(self.janela, text="A treinar a rede...", font=("Arial", 16))
        self.resultado.pack()

        self.canvas.bind("<Button-1>", self.pintar)
        self.canvas.bind("<B1-Motion>", self.pintar)
        self.canvas.bind("<ButtonRelease-1>", self.largar)
        tk.Button(self.janela, text="Classificar", command=self.classificar).pack(side="left", padx=10)
        tk.Button(self.janela, text="Limpar", command=self.limpar).pack(side="right", padx=10)
        self.janela.after(200, self.iniciar)

    def pintar(self, evento):
        x = evento.x
        y = evento.y
        if self.ultimo is not None:
            x0 = self.ultimo[0]
            y0 = self.ultimo[1]
            self.canvas.create_line(x0, y0, x, y, width=PINCEL, fill="black", capstyle="round")
            self.desenho.line([x0, y0, x, y], fill=255, width=PINCEL)
        self.ultimo = (x, y)

    def largar(self, evento):
        self.ultimo = None

    def classificar(self):
        if self.rede is None:
            return
        entrada = imagem_para_entrada(self.imagem)
        if entrada is None:
            self.resultado.config(text="Desenha um dígito")
            return

        saida = self.rede.propagar(entrada)
        texto_classes = ""
        for d in range(len(saida)):
            texto_classes += f"{d}:{saida[d]:+.2f}  "
        self.resultado.config(text=f"Dígito: {saida.index(max(saida))}\n{texto_classes}",
                              font=("Arial", 12))

    def limpar(self):
        self.desenho.rectangle([0, 0, TAM, TAM], fill=0)
        self.canvas.delete("all")
        if self.rede is not None:
            self.resultado.config(text="Desenha um dígito", font=("Arial", 16))

    def iniciar(self):
        self.rede = obter_rede()
        self.resultado.config(text="Desenha um dígito")

    def executar(self):
        self.janela.mainloop()


if __name__ == "__main__":
    AppDigitos().executar()
