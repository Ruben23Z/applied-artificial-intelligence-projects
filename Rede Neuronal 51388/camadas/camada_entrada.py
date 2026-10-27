"""
Camada de entrada (RNA Parte 2, slide 18, 34)
Regista e disponibiliza os dados de entrada, sem processamento:
    neurónios implementam a função identidade, y = f(x) = x.
Garante a mesma interface propagar(x) das restantes camadas, para a
RedeNeuronal encadear a propagação sem distinguir o tipo de camada.
"""
class CamadaEntrada:
    """Camada de entrada: disponibiliza x como y, sem transformação."""

    def __init__(self, ds):
        """
        Args:
            ds: dimensão de saída (número de saídas da camada, igual
                ao número de entradas da rede)
        """
        self.ds = ds
        self.y = [0 for _ in range(ds)]  # saída inicial, antes de propagar

    def propagar(self, x):
        """
        Propaga o vetor de entrada x, devolvendo-o inalterado como saída
        (função identidade: y = x).

        Args:
            x: vetor de entrada
        Returns:
            O próprio vetor x.
        """
        self.y = x  # mantém a informação de saída (f(xi) = xi)
        return self.y
