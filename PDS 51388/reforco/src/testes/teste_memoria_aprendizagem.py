from reforco.src.memorias.memoria_aprendizagem import MemoriaAprendizagem

"""
Testa a memória de aprendizagem, que guarda Q(s, a).

Nos testes o 'a' é uma ação, o segundo elemento do par (s, a) que indexa
Q(s, a). É só um nome qualquer, porque a memória aceita qualquer chave que
se possa usar num dicionário. No corredor seriam -1 (ESQUERDA) e 1 (DIREITA).

O valor por omissão é o Q(s, a) que q devolve para um par que ainda não foi
guardado com actualizar. Por defeito é 0.0, ou seja, o agente começa sem
preferência por nenhuma ação. Se for maior, por exemplo 5.0, as ações por
experimentar parecem melhores do que as já testadas, o que obriga o agente
a experimentá-las.
"""
def teste_memoria():
    m = MemoriaAprendizagem()
    print("q(1, 'a') sem entrada devolve o valor por omissão (esperado 0.0):", m.q(1, 'a'))
    print("q(1, 'a') com valor por omissão 5.0 (esperado 5.0):", MemoriaAprendizagem(5.0).q(1, 'a'))

    m.actualizar(1, 'a', 2.5)
    print("actualizar(1, 'a', 2.5) guarda, q(1, 'a') devolve (esperado 2.5):", m.q(1, 'a'))

    m.actualizar(1, 'a', 4.0)
    print("actualizar(1, 'a', 4.0) substitui, q(1, 'a') devolve (esperado 4.0):", m.q(1, 'a'))

    m.actualizar(2, 'b', 1.0)
    m.q(3, 'a')
    print("obter_estados devolve só os actualizados, ler o 3 com q não conta (esperado {1, 2}):", m.obter_estados())


if __name__ == "__main__":
    teste_memoria()

"""
Resultados
q(1, 'a') sem entrada devolveu 0.0, e com valor por omissão 5.0 devolveu 5.0.
Depois de actualizar(1, 'a', 2.5), q(1, 'a') devolveu 2.5. Depois de
actualizar(1, 'a', 4.0), devolveu 4.0. obter_estados devolveu {1, 2}, sem o
estado 3, que só foi lido com q.

Conclusões
Um par nunca guardado devolve o valor por omissão, que se escolhe ao criar a
memória. Guardar o mesmo par duas vezes deixa só o último valor. Só actualizar
regista o estado como observado, ler com q não regista. Isto interessa à
política, que só decide nos estados em que o agente já agiu.
"""