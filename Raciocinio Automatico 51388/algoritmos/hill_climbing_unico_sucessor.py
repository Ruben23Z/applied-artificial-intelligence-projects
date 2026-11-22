class HillClimbingUnicoSucessor:

    def __init__(self, max_tentativas=1000):
        self.max_tentativas = max_tentativas

    def iniciar(self, problema):
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)

        while True:
            melhorou = False
            for _ in range(self.max_tentativas):
                sucessor = problema.sucessor_aleatorio(self.estado)
                self.valor = problema.avaliar(sucessor)
                if self.valor > self.valor_estado:
                    self.estado = sucessor
                    self.valor_estado = self.valor
                    melhorou = True
                    break

            if not melhorou:
                break
        return self.estado
