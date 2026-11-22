import random


class HillClimbingEstocastico:

    def iniciar(self, problema):
        self.estado = problema.estado_inicial()
        self.valor_estado = problema.avaliar(self.estado)

        while True:
            melhores = []
            for sucessor in problema.sucessores(self.estado):
                self.valor = problema.avaliar(sucessor)
                if self.valor > self.valor_estado:
                    melhores.append((sucessor, self.valor))

            if not melhores:
                break

            self.estado, self.valor_estado = random.choice(melhores)

        return self.estado

