from camada_densa import CamadaDensa
from camada_entrada import CamadaEntrada

"""
Rede neuronal (RNA Parte 2, slides 1-21, 37-38)

Teorema da Aproximação Universal (slides 11-13): uma rede com uma
única camada escondida, com um número finito de neurónios, aproxima
qualquer função contínua num intervalo compacto, desde que φ seja
não linear, mas o teorema garante a existência dessa rede, não
indica a arquitetura nem os pesos concretos (problema da
aprendizagem, slide 13). Esta classe implementa apenas a propagação
direta (inferência com pesos fixos ou aleatórios), mas a determinação
dos pesos por treino ou retropropagação do erro não está implementada

Composta por uma camada de entrada seguida de uma ou mais camadas
densas (escondidas e a de saída), interligadas sequencialmente
(slide 15). 
O XOR não é resolúvel por uma única camada (perceptrão),
pois exige duas fronteiras de separação linear em simultâneo
(slides 3-6), em que as camadas escondidas resolvem isto projetando os
dados num espaço de maior dimensionalidade, onde se tornam
linearmente separáveis (slides 7-9). É esta transformação não
linear, repetida em cada camada, que dá à rede a capacidade de
representar relações complexas (slide 28)

Ativação direta da rede produz a saída y da rede com base na entrada X por ativação sequencial
encadeada das várias camadas da rede(slide 21):
    y¹ = x
    yⁿ = φⁿ(xⁿ·Wⁿ + bⁿ), para n > 1, com xⁿ = y^(n-1)
até se obter a saída final y^N. 
Os pesos e pendores de todas as camadas (exceto a de entrada) formam os parâmetros θ da rede(slide 20).

"""


class RedeNeuronal:
    """Rede neuronal multicamada, composta por uma camada de entrada, saída e camadas densas."""

    def __init__(self, forma, phi):
        """
        Args:
            forma: forma da rede — lista com o número de neurónios por
                   camada, incluindo a camada de entrada (forma[0])
            phi: função de ativação φ, comum aos neurónios das camadas
                 escondidas e de saída (não se aplica à camada de entrada,
                 que usa a função identidade)
        """
        self.forma = forma  # forma da rede (lista de n.º de neurónios por camada)
        self.phi = phi  # função de ativação dos neurónios das camadas
        self.camadas = []  # rede de camadas interligadas
        self.iniciar(forma, phi)

    def iniciar(self, forma, phi):
        """Constrói a camada de entrada e as camadas densas subsequentes,
        ligando a dimensão de saída de cada camada à dimensão de entrada
        da seguinte."""

        self.N = len(forma)  # número de camadas = profundidade da rede
        ds1 = forma[0]  # camada 1 (camada de entrada)
        camada = CamadaEntrada(ds1)
        self.camadas.append(camada)

        # Constrói as restantes camadas (escondidas e de saída) como camadas densas.
        # A dimensão de entrada de cada camada corresponde à dimensão de saída da
        # camada anterior (de n=1 até N-1), formando a cadeia de propagação.
        for n in range(1, self.N):
            de = forma[n - 1]
            ds = forma[n]
            camada = CamadaDensa(de, ds, phi)
            self.camadas.append(camada)

    def prever(self, X):
        """
        Prevê as saídas da rede para um conjunto de vetores de entrada,
        aplicando propagar(x) a cada um (diapositivo 38).
        Args:
            X: conjunto de vetores de entrada
        Returns:
            Conjunto de vetores de saída correspondentes (um por entrada).
        """
        Y = [self.propagar(x) for x in X]
        return Y

    def propagar(self, x):
        """
        Ativação direta da rede (diapositivo 21): propaga um vetor
        de entrada x sequencialmente por todas as camadas.
        A saída de cada camada (y^(n-1)) serve de entrada à camada
        seguinte (x^n), até se obter a saída final y^N da rede.
        Args:
            x: vetor de entrada (x¹ = x)
        Returns:
            Vetor de saída da rede (y^N).
        """
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y
