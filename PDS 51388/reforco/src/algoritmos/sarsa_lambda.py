"""
Algoritmo SARSA(lambda), Aprendizagem por Reforço Parte 4, slides 4 a 8

Cada par (s, a) tem um rasto e(s, a) que mede há quanto tempo e com que frequência foi visitado, e o erro de
cada passo atualiza todos os pares com rasto, ponderado por e (slide 6)
    delta = r + gama * Q(sn, an) - Q(s, a)
    Q(s, a) = Q(s, a) + alfa * delta * e(s, a)
    e(s, a) = gama * lambda * e(s, a)
Em estado terminal delta = r - Q(s, a), como o J = r do SARSA. Com lambda 0 só o
par atual tem rasto e o comportamento é o do SARSA. Com lambda perto de 1 o erro
chega a todo o percurso, mas pode reproduzir erros anteriores (slide 8)

Herda do Sarsa (on-policy) e só redefine aprender e episodio.
"""
from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.memorias.rastro_eligibilidade import RastroEligibilidade


class SarsaLambda(Sarsa):

    def __init__(self, mem_aprend, sel_accao, alfa, gama, lamb, modo):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.lamb = lamb
        self.rastos = RastroEligibilidade(modo)

    def aprender(self, s, a, r, sn, an=None):
        q = self.mem_aprend.q(s, a)
        if an is None:
            delta = r - q
        else:
            delta = r + self.gama * self.mem_aprend.q(sn, an) - q

        # visitar antes de atualizar e decair depois, senão o par atual perdia peso
        self.rastos.visitar(s, a)
        for (si, ai), e in self.rastos.pares():
            qi = self.mem_aprend.q(si, ai)
            self.mem_aprend.actualizar(si, ai, qi + self.alfa * delta * e)
        self.rastos.decair(self.gama * self.lamb)

    def episodio(self, ambiente):
        self.rastos.limpar()
        super().episodio(ambiente)
