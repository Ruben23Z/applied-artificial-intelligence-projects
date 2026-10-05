import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from Testes.testes_digitos import treinar_rede
from funcao_ativacao.tanh import Tanh

"""
Desenha um dígito à mão livre (arrastar o rato) e a rede classifica.
O desenho é feito em alta resolução e só depois reduzido para 8x8
(recorte, centragem e média por blocos), o que o aproxima das imagens de treino.
"""

camadas = [64, 32, 16, 10]
func_ativacao = Tanh()
epocas = 100
erro_max = 0.01
alpha = 0.01
beta = 0.5
n_treino = 500

TAM = 720  # lado do canvas em pixeis
PINCEL = 18  # espessura do traço

rede = treinar_rede(n_treino, camadas, func_ativacao, epocas, erro_max, alpha, beta)

imagem = Image.new("L", (TAM, TAM), 0)
desenho = ImageDraw.Draw(imagem)
ultimo = [None]  # último ponto do traço, para ligar com linhas

janela = tk.Tk()
janela.title("Dígitos")
canvas = tk.Canvas(janela, width=TAM, height=TAM, bg="white")
canvas.pack()
resultado = tk.Label(janela, text="Desenha um dígito", font=("Arial", 16))
resultado.pack()


def pintar(evento):
    x, y = evento.x, evento.y
    if ultimo[0] is not None:
        x0, y0 = ultimo[0]
        canvas.create_line(x0, y0, x, y, width=PINCEL, fill="black", capstyle="round")
        desenho.line([x0, y0, x, y], fill=255, width=PINCEL)
    ultimo[0] = (x, y)


# Para que o último local onde se desenhou não influencia o proximo desenho
def largar(evento):
    ultimo[0] = None


def imagem_para_entrada(img):
    """
    Converte uma imagem em tons de cinzento (traço claro sobre fundo preto)
    num vetor de 64 valores em [0, 1], parecido com as imagens de treino:
    dígito recortado, centrado, com uma pequena margem e reduzido para 8x8.
    """
    caixa = img.getbbox()  # retângulo que contém o traço
    if caixa is None:
        return None
    img = img.crop(caixa)

    # Coloca o recorte num quadrado, centrado, para não deformar o dígito
    lado = max(img.size)
    quadrado = Image.new("L", (lado, lado), 0)
    quadrado.paste(img, ((lado - img.width) // 2, (lado - img.height) // 2))

    # Margem à volta do dígito, como nas imagens do dataset
    quadrado = ImageOps.expand(quadrado, border=lado // 8, fill=0)

    # A média por blocos dá tons de cinzento, tal como nos dados de treino
    pequena = quadrado.resize((8, 8), Image.BOX)
    pixeis = list(pequena.getdata())
    maximo = max(pixeis)
    return [p / maximo for p in pixeis]  # o pixel mais forte fica a 1


def classificar():
    entrada = imagem_para_entrada(imagem)
    if entrada is None:
        resultado.config(text="Desenha um dígito")
        return
    saida = rede.propagar(entrada)
    resultado.config(text=f"Dígito: {saida.index(max(saida))}")


def limpar():
    desenho.rectangle([0, 0, TAM, TAM], fill=0)
    canvas.delete("all")
    resultado.config(text="Desenha um dígito")


canvas.bind("<Button-1>", pintar)
canvas.bind("<B1-Motion>", pintar)
canvas.bind("<ButtonRelease-1>", largar)
tk.Button(janela, text="Classificar", command=classificar).pack(side="left", padx=10)
tk.Button(janela, text="Limpar", command=limpar).pack(side="right", padx=10)
janela.mainloop()
