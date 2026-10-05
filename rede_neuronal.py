from camadas.camada_densa import CamadaDensa
from camadas.camada_entrada import CamadaEntrada

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



    def delta_saida(self, y_n, y):
        """
        Calcula a perda de saída da rede: δ^N = y^N - y (slide 41, parte 3),
        para a função de perda EQM (slide 28, parte 3).
        Args:
            y_n: vetor de saída gerado pela rede (y^N)
            y: vetor de treino de saída (alvo)
        Returns:
            Vetor de variação de erro de saída (δ^N)
        """
        return [y_n[k] - y[k] for k in range(len(y))]

    def retropropagar(self, delta_N, alpha, beta):
        """
        Retropropaga o erro pelas camadas da rede, exceto a camada de
        entrada (slide 42, parte 3).
        Args:
            delta_N: vetor de variação de erro de saída da rede (δ^N)
            alpha: taxa de aprendizagem (α)
            beta: fator de momento (β)
        """
        delta_n = delta_N  # δ^n começa como δ^N (erro da camada de saída)

        # Percorre as camadas de N até 2 (índice Python: de len-1 até 1),
        # nunca chegando à camada de entrada (índice 0), que apenas mantém
        # as entradas e não tem parâmetros a adaptar
        for n in range(len(self.camadas) - 1, 0, -1):
            yn_1 = self.camadas[n - 1].y  # saída da camada anterior (y^(n-1))
            dn_1 = self.camadas[n - 1].ds  # dimensão da camada n-1
            dn = self.camadas[n].ds  # dimensão da camada n
            neuronios_n = self.camadas[n].neuronios  # neurónios da camada n

            delta_n_1 = [0 for _ in range(dn_1)]  # δ^(n-1), um valor por neurónio de n-1

            # Para cada neurónio i da camada anterior (n-1), soma a
            # contribuição de todos os neurónios j da camada atual (n)
            for i in range(dn_1):
                soma = 0
                for j in range(dn):
                    soma = soma + (neuronios_n[j].w[i] * delta_n[j] * neuronios_n[j].y_derivada)
                delta_n_1[i] = soma

            # Adapta os pesos e pendores da camada n com o δ^n atual
            # (antes de ser substituído pelo δ^(n-1) calculado acima)
            self.camadas[n].adaptar(delta_n, yn_1, alpha, beta)
            delta_n = delta_n_1  # avança para a próxima iteração (camada n-1)

    def adaptar(self, x, y, alpha, beta):
        """
        Realiza um passo completo de aprendizagem para um par (entrada,
        saída) de treino: propagação direta, cálculo do erro de saída,
        retropropagação e cálculo do erro médio (slide 43, parte 3).
        Args:
            x: vetor de treino de entrada
            y: vetor de treino de saída (alvo)
            alpha: taxa de aprendizagem (α)
            beta: fator de momento (β)

        Returns:
            Erro médio de saída da rede (ε)
        """
        y_N = self.propagar(x)  # propagação direta, gera a saída da rede (y^N)
        delta_N = self.delta_saida(y_N, y)  # perda de saída (δ^N = y^N - y)
        self.retropropagar(delta_N, alpha, beta)  # retropropaga o erro e adapta os parâmetros

        K = len(delta_N)  # dimensão do vetor de perda
        erro = sum(delta_k ** 2 for delta_k in delta_N) / K  # erro médio de saída (ε), Erro Quadrático Médio

        return erro

    def treinar(self, X, Y, num_epocas, erro_max, alpha, beta):
        """
        Treina a rede neuronal por retropropagação do erro, ao longo de
        várias épocas, até o erro máximo ser atingido ou o número máximo
        de épocas ser alcançado (slide 44, parte 3).

        Args:
            X: conjunto de vetores de treino de entrada
            Y: conjunto de vetores de treino de saída (alvo)
            num_epocas: número máximo de épocas de treino
            erro_max: erro máximo aceite para finalizar o treino antecipadamente
            alpha: taxa de aprendizagem (α)
            beta: fator de momento (β)

        """
        for epoca in range(num_epocas):
            erro = 0
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                erro_x = self.adaptar(x, y, alpha, beta)  # erro de treino para o exemplo x
                erro = max(erro, erro_x)
            if erro <= erro_max:
                break
