"""
Modelo do mundo com predecessores, Aprendizagem por Reforço Parte 4, slide 18

O varrimento com prioridade faz a atualização retrospetiva. Quando o valor de
um estado muda, os pares (s, a) que levam a esse estado passam a ter mais
informação para propagar. Por isso o modelo guarda, para cada estado, os pares
que o têm como estado seguinte. Assume um ambiente determinista, e se um par
mudar de estado seguinte o predecessor antigo é retirado
"""
from reforco.src.memorias.modelo_tr import ModeloTR


class ModeloTRPred(ModeloTR):

    def __init__(self):
        super().__init__()
        self.pred = {}  # sn -> conjunto de pares (s, a)

    def actualizar(self, s, a, r, sn):
        antigo = self.T.get((s, a))
        if antigo is not None and antigo != sn:
            self.pred[antigo].discard((s, a))
        super().actualizar(s, a, r, sn)
        self.pred.setdefault(sn, set()).add((s, a))

    def predecessores(self, s):
        """Devolve os pares (s, a) já vividos que levaram ao estado s"""
        return self.pred.get(s, set())