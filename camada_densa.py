from neuronio import Neuronio

"""
Camada densa (RNA Parte 2, slides 19-31, 35-36).
Representa uma camada escondida ou de saída de uma rede multicamada.
Cada camada n é caracterizada (slide 19) por:
    - matriz de pesos W^n (ligações da camada anterior aos neurónios
      desta camada)
    - vetor de pendores b^n
    - função de ativação φ^n
    - vetor de saida y^n é criado aplicando a funcao de ativação à
    soma ponderada das entradas pelos respetivos pesos e somados os pendores

Saída: h^n = x^n·W^n + b^n (soma ponderada + pendor) ou para uma camada K-> hₖⁿ = Σⱼ₌₁ᵐ (xⱼⁿ · wⱼₖⁿ) + bₖⁿ;
y^n = φ^n(h^n) (slides 19-20), onde os pesos e pendores de todas as camadas (exceto a de
entrada) constituem os parâmetros θ da rede (slide 20).

Nesta implementação, em vez de uma única matriz W^n, cada Neuronio
guarda o seu próprio vetor de pesos e pendor, equivalente a uma
coluna de W^n e à componente k de b^n, em que propagar(x) invoca
em cada neurónio, o que é matematicamente equivalente ao
produto matricial x^n·W^n + b^n seguido de φ^n (ver exemplo REDE XOR,
slides 22-27, onde os pesos são atribuídos neurónio a neurónio).

É a organização em camadas escondidas que dá à rede a capacidade de separação não linear
e as suas características principais (slides 28-29), como a extração
progressiva de características mais abstratas, tolerância a ruído,
processamento distribuído e escalabilidade mas à custa de maior
dificuldade de interpretação e necessidade de mais dados de treino
(slides 30-31).
"""


class CamadaDensa:
    """Camada densa: conjunto de neurónios totalmente ligados às entradas."""

    def __init__(self, de, ds, phi):
        """
        Inicializa a camada com os respetivos neurónios.

        Args:
            de: dimensão de entrada (número de ligações de entrada,
                igual à dimensão de saída da camada anterior)
            ds: dimensão de saída (número de neurónios da camada; cada
                neurónio produz uma componente do vetor y^n)
            phi: função de ativação φ^n, comum a todos os neurónios
                 da camada
        """
        self.de = de
        self.ds = ds
        self.phi = phi
        self.y = []  # saída y^n da camada. Só é preenchida após a primeira chamada a propagar()
        self.neuronios = [Neuronio(de, phi) for _ in range(ds)]  # um neurónio por saída (coluna de W^n)

    def propagar(self, x):
        """
        Propaga o vetor de entrada x por cada neurónio da camada,
        calculando y^n = φ^n(x^n·W^n + b^n) de forma distribuída
        (um neurónio por componente da saída).

        Args:
            x: vetor de entrada (x^n = y^(n-1), saída da camada anterior)
        Returns:
            Lista com a saída de cada neurónio da camada (vetor y^n).
        """
        self.y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return self.y

    def adapta(self, delta, yn_1, alpha):
        """
        Adapta os pesos e pendores dos neurónios da camada (slide 40, Parte 3).
        Args:
            delta: vetor de erro de saída da camada (δ^n), um valor por neurónio
            yn_1: vetor de saída da camada anterior (y^(n-1)), entrada comum a todos os neurónios
            alpha: taxa de aprendizagem (α)
        """

        for j in range(self.ds):
            self.neuronios[j].adaptar(delta[j], yn_1, alpha)
