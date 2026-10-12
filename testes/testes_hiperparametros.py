import numpy as np
from funcao_ativacao.tanh import Tanh
from funcao_ativacao.sigmoid import Sigmoid
from rede_neuronal import RedeNeuronal
from utils import graficos
from utils.utilidades import treinar_com_historico

"""Testa diferentes valores dos hiperparâmetros da rede neuronal e compara
os resultados. Cada valor é treinado 20 vezes (sementes 0 a 19, iguais para
todos os valores, para a comparação ser justa). Mostra média ± desvio padrão
e mediana.
-----------------------------------------------------

Para 1 teste dos Hiperparâmetro
Condições: 20 repetições (sementes 0 a 19), 1000 épocas, erro máximo 0.05.
Valores-base: 3 escondidos, alpha = 0.2, beta = 0.0, tanh.

alpha (convergiram):
    0.01: 10/20 | 0.05: 20/20 | 0.1: 20/20 | 0.2: 20/20 | 0.5: 20/20
    Mediana de épocas: 676, 196, 104, 60, 57.
beta (convergiram):
    0.0: 20/20 | 0.1: 20/20 | 0.5: 20/20 | 0.9: 1/20 | 2.0: 0/20
    Mediana de épocas: 60, 54, 42, 44, -.
escondidos (convergiram):
    1: 0/20 | 2: 8/20 | 3: 20/20 | 5: 20/20 | 9: 20/20
    Mediana de épocas: -, 79, 60, 36, 30.
phi (convergiram):
    Tanh: 20/20, mediana 60 | Sigmoid: 13/20, mediana 744.
    
Conclusões:
- Alpha: o número de épocas diminui com o aumento de alpha. Com 0.01, 10 redes em 20 não convergiram em 1000 épocas.
- Beta: até 0.5 o número de épocas diminui. Com 0.9 convergiu 1 rede em 20 e com 2.0 nenhuma.
- Neurónios escondidos: com 1, nenhuma rede convergiu e com 2 convergiram 8 em 20. Com 3 ou mais convergiram todas, e a mediana das épocas
diminui com o aumento do número de neurónios.
- Função de ativação: a tanh convergiu em 20 redes e a sigmoide em 13.
A mediana de épocas foi 60 com tanh e 744 com sigmoide.
"""


class TesteHiperparametros:

    def __init__(self, entradas, saidas):
        self.entradas = entradas
        self.saidas = saidas
        self.repeticoes = 20
        self.epocas = 1000
        self.erro_max = 0.05

        # Em cada teste, apenas um destes valores é alterado
        self.base = {"escondidos": 3, "alpha": 0.2, "beta": 0.0, "phi": Tanh()}

    def treinar(self, escondidos=3, alpha=0.2, beta=0.0, phi=None):
        if phi is None:
            phi = Tanh()

        forma = [len(self.entradas[0]), escondidos, len(self.saidas[0])]
        rede = RedeNeuronal(forma, phi)

        try:
            epoca, erro, historico = treinar_com_historico(rede, self.entradas, self.saidas,
                                                           self.epocas, self.erro_max, alpha, beta)
        except OverflowError:
            # considera-se que a rede divergiu
            return self.epocas, float("inf"), False, [float("inf")]

        return epoca, erro, erro <= self.erro_max, historico

    def curva_completa(self, historico):
        """Devolve o erro de todas as épocas (limitado a 1). As redes que
        pararam cedo ficam com o último erro até ao fim."""
        curva = []
        for e in range(self.epocas):
            if e < len(historico):
                curva.append(min(historico[e], 1.0))
            else:
                curva.append(min(historico[-1], 1.0))
        return curva

    def testar(self, parametro, valores):
        print(f"\nExploração de {parametro} ({self.repeticoes} repetições)")
        print("Valor | Convergiu | Épocas conv. (média ± desvio padrão) | Mediana | Erro final (média ± desvio padrão)")

        curvas = {}
        taxas = []
        nomes = []

        for valor in valores:
            config = self.base.copy()
            config[parametro] = valor

            epocas_conv = []  # épocas só das redes que convergiram
            erros = []
            curvas_rede = []
            convergencias = 0

            for semente in range(self.repeticoes):
                np.random.seed(semente)
                epoca, erro, convergiu, historico = self.treinar(**config)

                if convergiu:
                    convergencias += 1
                    epocas_conv.append(epoca)
                erros.append(min(erro, 1.0))
                curvas_rede.append(self.curva_completa(historico))

            if parametro == "phi":
                nome = type(valor).__name__
            else:
                nome = str(valor)

            if len(epocas_conv) > 0:
                texto_epocas = f"{np.mean(epocas_conv):.1f} ± {np.std(epocas_conv):.1f} | {np.median(epocas_conv):.0f}"
            else:
                texto_epocas = "- | -"
            print(f"{nome} | {convergencias}/{self.repeticoes} | {texto_epocas} | "
                  f"{np.mean(erros):.4f} ± {np.std(erros):.4f}")

            curvas[nome] = np.median(curvas_rede, axis=0)
            nomes.append(nome)
            taxas.append(convergencias / self.repeticoes)

        graficos.plot_curvas(curvas, f"Erro por época (mediana) - {parametro}", "Época", "Erro máximo",
                             f"hiper_{parametro}.png", log=True)
        if parametro == "escondidos":
            graficos.plot_barras(nomes, {"taxa": taxas}, "Taxa de convergência vs neurónios escondidos",
                                 "Fração de redes que convergiram", "hiper_escondidos_taxa.png")

    def executar(self):
        self.testar("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
        self.testar("beta", [0.0, 0.1, 0.5, 0.9, 2.0])
        self.testar("escondidos", [1, 2, 3, 5, 9])
        self.testar("phi", [Tanh(), Sigmoid()])


entradas = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
]

saidas = [[0], [1], [1], [1], [0], [0], [0]]

if __name__ == "__main__":
    teste = TesteHiperparametros(entradas, saidas)
    teste.executar()
