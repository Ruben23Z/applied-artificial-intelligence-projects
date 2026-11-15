import random

from reforco.src.algoritmos.sarsa import Sarsa
from reforco.src.algoritmos.sarsa_lambda import SarsaLambda
from reforco.src.corredor.ambiente_corredor import AmbienteCorredor
from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem
from reforco.src.memorias.rastro_eligibilidade import RastroEligibilidade
from reforco.src.sel_accao.egreedy import EGreedy


def novo(lamb, modo='acumulacao'):
    memoria = MemoriaAprendizagem()
    sel = EGreedy(memoria, ['a'], 0)
    return memoria, SarsaLambda(memoria, sel, 0.5, 0.9, lamb, modo)


def teste_aprender():
    """
    Testa o passo do SARSA(lambda) (Parte 4, slides 5 a 7)
    """
    memoria, sl = novo(0.9)
    sl.aprender('s', 'a', 1.0, 'sn')
    print("Passo terminal. Esperado: 0.5. Obtido:", memoria.q('s', 'a'))

    memoria, sl = novo(0.9)
    sl.aprender('s', 'a', 0.0, 'sn', 'an')
    sl.aprender('sn', 'an', 1.0, 't')
    print("Dois passos Q(sn, an). Esperado: 0.5. Obtido:", memoria.q('sn', 'an'))
    print("Dois passos Q(s, a). Esperado: 0.405. Obtido:", memoria.q('s', 'a'))

    memoria, sl = novo(0.0)
    sl.aprender('s', 'a', 0.0, 'sn', 'an')
    sl.aprender('sn', 'an', 1.0, 't')
    print("Com lambda 0 Q(s, a). Esperado: 0.0. Obtido:", memoria.q('s', 'a'))


def teste_rastros():
    for modo, esperado in (('acumulacao', 1.81), ('substituicao', 1.0)):
        rastro = RastroEligibilidade(modo)
        rastro.visitar('s', 'a')
        rastro.decair(0.81)
        rastro.visitar('s', 'a')
        print("Rastro", modo, "| Esperado:", esperado, "| Obtido:", round(rastro.e[('s', 'a')], 3))


def episodios_ate_valor(fabrica, semente, alvo=0.7, max_episodios=500):
    random.seed(semente)
    ambiente = AmbienteCorredor()
    memoria = MemoriaAprendizagem()
    sel = EGreedy(memoria, ambiente.accoes(), 0.1)
    algoritmo = fabrica(memoria, sel)
    for n in range(1, max_episodios + 1):
        algoritmo.episodio(ambiente)
        if memoria.q(4, AmbienteCorredor.DIREITA) >= alvo:
            return n
    return max_episodios


def teste_corredor(numero_sementes=100):
    casos = (("SARSA", lambda m, s: Sarsa(m, s, 0.1, 0.9)),
             ("lambda 0.5", lambda m, s: SarsaLambda(m, s, 0.1, 0.9, 0.5, 'acumulacao')),
             ("lambda 0.9", lambda m, s: SarsaLambda(m, s, 0.1, 0.9, 0.9, 'acumulacao')))
    for nome, fabrica in casos:
        total = sum(episodios_ate_valor(fabrica, k) for k in range(numero_sementes))
        print(nome, "| média de episódios até Q(4, DIREITA) >= 0.7:", round(total / numero_sementes, 2))


if __name__ == "__main__":
    teste_aprender()
    teste_rastros()
    teste_corredor()


"""
Resultados e conclusões

Passo terminal. Esperado: 0.5. Obtido: 0.5
Dois passos Q(sn, an). Esperado: 0.5. Obtido: 0.5
Dois passos Q(s, a). Esperado: 0.405. Obtido: 0.405
Com lambda 0 Q(s, a). Esperado: 0.0. Obtido: 0.0
Rastro acumulacao | Esperado: 1.81 | Obtido: 1.81
Rastro substituicao | Esperado: 1.0 | Obtido: 1
SARSA | média de episódios até Q(4, DIREITA) >= 0.7: 58.59
lambda 0.5 | média de episódios até Q(4, DIREITA) >= 0.7: 41.72
lambda 0.9 | média de episódios até Q(4, DIREITA) >= 0.7: 25.11

No passo terminal delta = r e Q(s, a) = 0.5, como no SARSA
Com dois passos e lambda 0.9 o rasto de (s, a) é 0.81 quando chega a recompensa, por isso o erro
propaga-se para trás e Q(s, a) = 0.5 * 1 * 0.81 = 0.405 
Com lambda 0 o rasto desaparece e Q(s, a) fica 0, ou seja, o comportamento é o do SARSA
A acumulação soma 1 a cada visita (1.81) e a substituição repõe a 1 
No corredor a média de episódios desce com o lambda, de 58.59 no SARSA para 41.72 com
lambda 0.5 e 25.11 com 0.9, porque cada recompensa atualiza todo o percurso
recente e o valor chega mais depressa ao estado 4
"""