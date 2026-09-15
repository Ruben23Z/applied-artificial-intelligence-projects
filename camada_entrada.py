class CamadaEntrada:
    """Camada de entrada, regista e disponibiliza os dados de entrada."""

    def __init__(self, ds):
        """
        Args:
            ds: dimensão de saída (número de saídas da camada)
        """
        self.ds = ds
        self.y = [0 for _ in range(ds)]

    def propagar(self, x):
        """
        Propaga o vetor de entrada x, gerando a saída da camada.

        Args:
            x: vetor de entrada
        """
        self.y = x  # mantém a informação de saída
        return self.y
