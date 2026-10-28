from reforco.src.algoritmos.sarsa import Sarsa


class QLearning(Sarsa):
    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        super().__init__(mem_aprend, sel_accao, alfa, gama)

    # a minha duvida é que o prof nos slides, algumas funcoes não tem o final, e ao seguir os slides supostamente não
    # deve ser cocoado o final. depois no q learning não ha verificacao se an é terminal e nem se usa o final. e
    # ainda nao sei se o final tem valor por omissao
    def aprender(self, s, a, r, sn, final):
        # aqui finalpode ser none, faz sentido? depois tudo dos outros algoritmos precisam de final

        an = self.sel_accao.accao_sofrega(sn)

        super().aprender(s, a, r, sn, an)

    def episodio(self, ambiente):
        s = ambiente.iniciar()
        terminou = False

        while not terminou:
            a = self.sel_accao.seleccionar_accao(s)
            sn, r, terminou = ambiente.executar(a)
            self.aprender(s, a, r, sn, terminou)
            s = sn
