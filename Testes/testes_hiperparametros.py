from funcao_ativacao.tanh import Tanh
from funcao_ativacao.sigmoid import Sigmoid
from rede_neuronal import RedeNeuronal

"""Classe usada para testar diferentes valores dos hiperparâmetros 
da rede neuronal e comparar os resultados obtidos."""


class TesteHiperparametros:

    def __init__(self, entradas, saidas):
        self.entradas = entradas
        self.saidas = saidas

        self.repeticoes = 5

        self.epocas = 1000

        self.erro_max = 0.05

        # Valores-base dos hiperparâmetros
        # Em cada teste, apenas um destes valores é alterado
        self.base = {"escondidos": 3, "alpha": 0.2, "beta": 0.0, "phi": Tanh()}

    def treinar(self, escondidos=3, alpha=0.2, beta=0.0, phi=None):
        if phi is None:
            phi = Tanh()

        # Define o número de entradas, neurónios escondidos e saídas
        forma = [len(self.entradas[0]), escondidos, len(self.saidas[0])]
        rede = RedeNeuronal(forma, phi)

        try:
            # Treina a rede até atingir o número máximo de épocas
            for epoca in range(1, self.epocas + 1):

                # Calcula o maior erro entre todos os exemplos
                erro = max(
                    rede.adaptar(x, y, alpha, beta)
                    for x, y in zip(self.entradas, self.saidas)
                )

                # Se o erro for suficientemente baixo, considera-se que a rede convergiu
                if erro <= self.erro_max:
                    return epoca, erro, True

        # Evita que o programa termine se os valores
        # provocarem uma instabilidade numérica
        except OverflowError:
            return self.epocas, float("inf"), False

        # Se não convergir dentro das épocas máximas
        return self.epocas, erro, False

    def testar(self, parametro, valores):

        print("\nExploração de", parametro)
        print("Valor | Convergiu | Época média | Erro médio")

        for valor in valores:

            config = self.base.copy()
            # Altera apenas o parâmetro que queremos testar
            config[parametro] = valor

            resultados = []

            # Treina várias redes para o mesmo valor
            for _ in range(self.repeticoes):
                resultados.append(self.treinar(**config))
            # Nº de redes a convergir
            convergencias = sum(r[2] for r in resultados)
            media_epocas = (sum(r[0] for r in resultados) / self.repeticoes)
            # média do erro final
            media_erro = (sum(r[1] for r in resultados) / self.repeticoes)

            # Para as funções de ativação, mostra o nome em vez da representação do objeto
            if parametro == "phi":
                valor = type(valor).__name__

            print(
                valor,
                "|", convergencias, "/", self.repeticoes,
                "|", round(media_epocas, 1),
                "|", round(media_erro, 5)
            )

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
    [0, 0, 0, 0, 0, 0, 1, 1, 1]
]

saidas = [
    [0],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0]
]

if __name__ == "__main__":
    teste = TesteHiperparametros(entradas, saidas)
    teste.executar()
