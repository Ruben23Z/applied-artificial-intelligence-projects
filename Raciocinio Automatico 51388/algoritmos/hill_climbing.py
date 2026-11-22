class HillClimbing:

    def iniciar(self, problema):
        self.estado = problema.estado_inicial()
        self.valor_estado = float("-inf")
        while (True):
            self.sucessores = problema.sucessores(self.estado)
            self.novo_estado = max(self.sucessores, key=problema.avaliar, default=None)
            if self.novo_estado is not None:
                self.valor_novo_estado = problema.avaliar(self.novo_estado)

            if self.valor_novo_estado <= self.valor_estado:
                self.valor_novo_estado <= self.valor_estado
                break

            self.estado = self.novo_estado
            self.valor_estado = self.valor_novo_estado

        return self.estado
