from camada_densa import CamadaDensa
from camada_entrada import CamadaEntrada


class RedeNeuronal:
    """Rede neuronal multicamada, composta por uma camada de entrada e camadas densas."""

    def __init__(self, forma, phi):
        """
        Args:
            forma: forma da rede (lista com o número de neurónios por camada)
            phi: função de ativação dos neurónios das camadas
        """
        self.forma = forma  # forma da rede (lista de n.º de neurónios por camada)
        self.phi = phi  # função de ativação dos neurónios das camadas
        self.camadas = []
        self.iniciar(forma, phi)

    def iniciar(self, forma, phi):
        """Constrói a camada de entrada e as camadas densas subsequentes."""
        self.N = len(forma)  # número de camadas        
        ds1 = forma[0]  # camada 1(camada de entrada)
        camada = CamadaEntrada(ds1)
        self.camadas.append(camada)

        # Constrói as restantes camadas, escondidas e de saída.
        # A dimensão de entrada de cada camada corresponde à dimensão de saída da camada anterior (de n=1 até N-1)
        for n in range(1, self.N):
            de = forma[n - 1]
            ds = forma[n]
            camada = CamadaDensa(de, ds, phi)
            self.camadas.append(camada)

    def prever(self, X):
        """
        Prevê as saídas da rede para um conjunto de vetores de entrada.

        Args:
            X: conjunto de vetores de entrada

        Returns:
            Conjunto de vetores de saída correspondentes.
        """
        Y = [self.propagar(x) for x in X]
        return Y

    def propagar(self, x):
        """
        Propaga um vetor de entrada x sequencialmente por todas as camadas.

        A saída de cada camada serve de entrada à camada seguinte, até se
        obter a saída final da rede.

        Args:
            x: vetor de entrada

        Returns:
            Vetor de saída da rede.
        """
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y



#faz e etstar com modulo de testes, com o teste do xor, experimentar valores aleatorios e comparar com os slides das aulas.
#primeira parte somente a da aula um, depois a parte dois da parte 1 é da aula de hoje
