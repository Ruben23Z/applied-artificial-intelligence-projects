from neuronio import Neuronio


class CamadaDensa:
    """Camada densa: conjunto de neurónios totalmente ligados às entradas."""

    def __init__(self, de, ds, phi):
        """
        Inicializa a camada com os respetivos neurónios.

        Args:
            de: dimensão de entrada (número de ligações de entrada)
            ds: dimensão de saída (número de neurónios/saídas da camada)
            phi: função de ativação dos neurónios
        """
        self.de = de
        self.ds = ds
        self.phi = phi
        self.y = []  # saída da camada. Só é preenchida após a primeira chamada a propagar()
        self.neuronios = [Neuronio(de, phi) for _ in range(ds)]

    def propagar(self, x):
        """
        Propaga o vetor de entrada x por cada neurónio da camada.

        Args:
            x: vetor de entrada

        Returns:
            Lista com a saída de cada neurónio da camada.
        """
        self.y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return self.y
