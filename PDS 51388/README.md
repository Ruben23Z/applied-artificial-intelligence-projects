# Matéria, Processos de Decisão Sequencial e Aprendizagem por Reforço Parte 1

Resumo da matéria dos dois PDFs de Luís Morgado, ISEL-DEI.

* Processos de Decisão Sequencial, referido como PDS
* Aprendizagem por Reforço, Parte 1, referido como APF1

Notação usada nas fórmulas

| Símbolo | Significado |
|---|---|
| s, sn | estado atual e estado seguinte (s' nos slides) |
| a, an | ação atual e ação seguinte (a' nos slides) |
| r | recompensa |
| gama | fator de desconto, entre 0 e 1 |
| alfa | fator de aprendizagem, entre 0 e 1 |
| epsilon | probabilidade de exploração, entre 0 e 1 |
| T(s, a, sn) | probabilidade de transição |
| R(s, a, sn) | recompensa de transição |
| U(s) | utilidade do estado s |
| Q(s, a) | estimativa do valor da ação a no estado s |
| pi | política |

## Índice

* Parte 1, Processos de Decisão Sequencial
* Parte 2, Aprendizagem por Reforço
* Ligação entre as duas partes

---

# Parte 1, Processos de Decisão Sequencial

## 1. Os dois problemas da inteligência artificial

* **Problema da classificação**, determinar o que é. Consiste em discriminar informação nos dados processados.
* **Problema da decisão**, determinar o que fazer. Consiste em decidir qual das opções de ação possíveis realizar.

No caso geral não são conhecidas soluções algorítmicas para estes problemas. As soluções são produzidas por aprendizagem, em particular por redes neuronais artificiais e por aprendizagem por reforço.

## 2. Tomada de decisão

A tomada de decisão é o processo cognitivo que resulta na seleção de uma opção entre várias alternativas. Baseia-se em alguma forma de valor ou preferência, que reflete o propósito, ou seja, o objetivo.

Há duas famílias de teorias.

| Racionalidade ilimitada | Racionalidade limitada |
|---|---|
| capacidade de representação e cálculo ilimitada | capacidade de representação e cálculo limitada |
| representações completas e consistentes | utilização de heurísticas |
| otimização | soluções aproximadas |

## 3. O conceito de agente

Um agente é um sistema computacional capaz de perceção, decisão e ação para a concretização autónoma de objetivos, maximizando uma medida de desempenho. Envolve cognição e racionalidade.

O agente opera num ambiente. Num instante t executa a ação a(t) e observa o estado s(t+1) e a recompensa r(t+1).

```
Agente    envia ao Ambiente      a ação a(t)
Ambiente  devolve ao Agente      o estado s(t+1) e a recompensa r(t+1)
```

## 4. Tomada de decisão sequencial

Num processo de decisão geral, para cada estado, cada decisão de ação (a1, a2, ..., ai) pode levar a várias ramificações de estados e decisões futuras.

* O valor, ou utilidade, desses estados e decisões pode não ser conhecido de imediato. Acontece diferido no tempo, em função dos encadeamentos de estados e decisões futuras.
* A própria evolução dos estados observados em função das ações realizadas pode não ser determinista, ou seja, pode estar sujeita a incerteza.

A pergunta é como decidir qual a ação a realizar. A utilidade é o valor a longo prazo.

## 5. Incerteza

A incerteza resulta da impossibilidade de obter informação completa relativa ao domínio do problema. O exemplo do slide é a navegação em veículos autónomos.

## 6. Processos de decisão sequencial

O problema da decisão ao longo do tempo tem as seguintes características.

* O valor de uma ação depende de uma sequência de decisões.
* Há possibilidade de ganhos e perdas.
* Há incerteza na decisão.
* O efeito é cumulativo.

O resultado de uma ação pode não ser completamente previsível, o que se designa por não determinismo da ação. A este tipo de processos correspondem espaços de estados não deterministas, em que as transições entre estados ocorrem por efeito das ações, com uma determinada probabilidade de transição. O sistema não tem controlo total sobre estas transições.

A cada transição pode estar associada uma recompensa, que representa o ganho ou a perda dessa transição de estado.

Exemplo do slide, com ações a0 e a1 e estados s0, s1 e s2. Cada transição tem uma probabilidade e, quando aplicável, um ganho ou uma perda.

## 7. Propriedade de Markov

Andrey Markov foi um matemático russo (1856 a 1922).

Um processo estocástico tem a propriedade de Markov se a distribuição probabilística condicional dos estados futuros depender exclusivamente do estado presente. A previsão dos estados seguintes só depende do estado presente.

```
... s(t), a(t), r(t+1), s(t+1), a(t+1), r(t+2), s(t+2), a(t+2), ...
```

Esta sequência de evolução ao longo dos estados possíveis, com estado, ação e recompensa, é uma cadeia de Markov.

## 8. Processo de decisão de Markov (PDM)

Um PDM é definido por cinco componentes, que formam o modelo do mundo.

| Componente | Descrição |
|---|---|
| Conjunto de estados S | estados possíveis em que o agente pode estar |
| Conjunto de ações A(s) | ações que o agente pode realizar em cada estado s |
| Função de transição T(s, a, sn), valor entre 0 e 1 | probabilidade de, a partir do estado s, transitar para sn ao realizar a |
| Função de recompensa R(s, a, sn) | recompensa por transitar de s para sn ao realizar a |
| Fator de desconto gama, valor entre 0 e 1 | determina a importância das recompensas futuras em relação às imediatas |

### Exemplo, ambiente determinista

Sete posições em fila.

```
1  2  3  4  5  6  7
-                 +
perda -1          ganho +1
```

```
S = {1, 2, 3, 4, 5, 6, 7}
A(s) = {esquerda, direita}, para todo o s em S

T(s, a, sn) = 1 se s em {2, 3, 4, 5, 6}    (sn é a posição a que a ação leva)
T(s, a, sn) = 0 se s em {1, 7}

R(s, a, sn) = 0   se s em {2, 3, 4, 5, 6}
R(s, a, sn) = +1  se sn = 7
R(s, a, sn) = -1  se sn = 1
```

## 9. Recompensa

A recompensa é uma medida de desempenho para a tomada de decisão que reflete o efeito cumulativo da evolução de estado através das recompensas observadas, expressando ganhos e perdas.

* É um valor finito, positivo (ganho) ou negativo (perda).
* Expressa o ganho ou a perda num determinado estado.
* Tem âmbito local, ou seja, de curto prazo.

Há três formas, consoante a dependência.

```
R(s)          depende apenas do estado
R(s, a)       depende do estado e da ação realizada
R(s, a, sn)   depende do estado, da ação e do estado seguinte
```

## 10. Retorno

O retorno G(t) expressa o efeito cumulativo das recompensas. Há três tipos.

```
Horizonte finito
    G(t) = r(t) + r(t+1) + ... + r(t+n)

Horizonte infinito
    G(t) = limite, com n a tender para infinito, de (1/n) * (r(t) + ... + r(t+n))

Horizonte infinito com desconto temporal, com gama entre 0 e 1
    G(t) = r(t) + gama * r(t+1) + gama^2 * r(t+2) + gama^3 * r(t+3) + ...
```

## 11. Utilidade (valor) de estado

A utilidade de um estado corresponde ao retorno esperado a partir desse estado. É o efeito cumulativo da evolução de estado através das recompensas observadas. Tem âmbito global, ou seja, de longo prazo. Representa o valor que pode ser obtido a partir de um estado ao longo de uma sequência de evolução de estado, e o valor do estado para a realização do objetivo pretendido.

A recompensa tem âmbito local e a utilidade tem âmbito global.

### Soma das recompensas

Se a recompensa depende apenas do estado,

```
U(s(t)) = R(s(t)) + R(s(t+1)) + R(s(t+2)) + ...
```

Esta forma tem dois problemas.

* As recompensas têm de estar limitadas a uma gama finita de valores, caso contrário a soma pode ser infinita, o que impossibilita a decisão.
* Não reflete o efeito da passagem do tempo. As recompensas mantêm o seu valor independentemente do momento em que ocorrem.

### Recompensas descontadas

```
U(s(t)) = R(s(t)) + gama * R(s(t+1)) + gama^2 * R(s(t+2)) + ...
```

O fator de desconto gama, entre 0 e 1, tem estas propriedades.

* Produz uma redução de valor não linear em função da distância no tempo da recompensa. A função gama^n é de decaimento exponencial em n.
* Reflete o efeito da passagem do tempo.
* O momento em que a recompensa ocorre determina a sua contribuição para a utilidade a longo prazo.
* Com desconto, a soma deixa de depender de as recompensas estarem limitadas a uma gama finita de valores.

Um gama perto de 0 dá mais peso às recompensas imediatas. Um gama perto de 1 dá mais peso às recompensas futuras. O slide de exemplo compara o efeito do fator de desconto com gama 0.9 e gama 0.5.

## 12. Política comportamental

A política representa o comportamento do agente. Define qual a ação que deve ser realizada em cada estado, ou seja, a estratégia de ação.

| Política determinista | Política não determinista |
|---|---|
| pi, função de S em A(s), para s em S | pi, função de S x A(s) em [0, 1], para s em S |
| para cada estado indica uma ação específica | para cada par (estado, ação) indica a probabilidade de realizar a ação nesse estado |
| pi(s1) = a1 | pi(s1, a1) = 0.2, pi(s1, a2) = 0.5, pi(s1, a3) = 0.3 |

Exemplo do slide. Utilidade de estado para um ambiente de 4 por 3, com gama igual a 1 e R(s) igual a -0.04 nos estados não terminais, e a política ótima correspondente à utilidade calculada. Os estados terminais têm recompensa +1 e -1.

## 13. Resolução de processos de decisão de Markov

O objetivo é encontrar uma política que maximize o retorno em cada estado, ou seja, a soma das recompensas esperadas ao longo do tempo.

Há duas vertentes na resolução.

* Cálculo da utilidade de estado.
* Cálculo da política de tomada de decisão.

## 14. Utilidade com não determinismo da ação

Com ações não deterministas é necessário considerar a probabilidade de ocorrência de cada valor possível de recompensa. A recompensa r(t+1) pode ter valores diferentes, consoante o efeito não determinista da ação a(t). Uma execução da ação a(t) no estado s(t) leva a transições de estado diferentes, com recompensas r1, r2, ..., rn, cada uma com a sua probabilidade.

Para considerar a probabilidade de cada valor, calcula-se o valor esperado, que pesa cada valor pela probabilidade da sua ocorrência.

```
E(x) = soma, para i de 0 a n, de p(xi) * xi
```

A utilidade de estado é o valor esperado das recompensas ao longo do tempo, descontadas em função do tempo.

```
U(s(t)) = E[ r(t+1) + gama * r(t+2) + gama^2 * r(t+3) + ... ]
```

Com um modelo de transição T(s, a, sn) e um modelo de recompensa R(s, a, sn), em que R(s(t), a(t), s(t+1)) pode gerar resultados diferentes com probabilidades diferentes,

```
U(s(t)) = E[ R(s(t), a(t), s(t+1)) + gama * R(s(t+1), a(t+1), s(t+2)) + ... ]
        = E[ R(s(t), a(t), s(t+1)) + gama * U(s(t+1)) ]
```

Para uma política pi, com m ações possíveis em s(t) e n estados seguintes possíveis,

```
U(s) = soma sobre i de 1 a m de pi(s, ai)
         * soma sobre j de 1 a n de T(s, ai, sn_j) * ( R(s, ai, sn_j) + gama * U(sn_j) )
```

## 15. O princípio da decisão ótima e a equação de Bellman

A decisão ótima é a que maximiza a utilidade de cada estado observado.

O cálculo da utilidade é possível por programação dinâmica, em que a solução é obtida por decomposição de um problema em subproblemas. Num PDM isto é viável devido à propriedade de Markov. As utilidades dos estados podem ser determinadas em função das utilidades dos estados sucessores. Esta relação é a equação de Bellman.

```
Utilidade de estado para uma política pi
    U(s) = soma sobre a de pi(s, a) * soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) )

Utilidade de estado para a política ótima pi*
    U*(s) = max sobre a de soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U*(sn) )

Política ótima pi*
    pi*(s) = argmax sobre a de soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) )
```

Para cada estado s, escolhe-se a ação a com maior utilidade.

## 16. Programação dinâmica

Resolve um problema por composição iterativa de soluções parciais. A iteração propaga retornos parciais e usa uma estimativa dos retornos futuros para estimar o retorno do estado atual.

```
U(s) = max sobre a de soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) )
```

Do estado s, cada ação a leva aos estados seguintes sn, com probabilidade T(s, a, sn) e recompensa R(s, a, sn), e o valor U(sn) já estimado de cada um entra no cálculo de U(s). O slide seguinte mostra o cálculo iterativo da utilidade para alguns estados do problema de 4 por 3 (Russell e Norvig, 2010).

## 17. Convergência da utilidade e da política

O erro da política, ou seja, o número de ações incorretas por estado, decresce mais depressa do que o erro da utilidade.

A política reflete o gradiente da função de utilidade, por isso tem um carácter relativo. Isto permite a convergência para a política ótima antes da convergência da função de utilidade. O gráfico do slide mostra o erro máximo e a perda de política em função do número de iterações.

## 18. Iteração da utilidade de estado

```
Iniciar
    U(s) = 0, para todo o s em S

Iterar
    U(s) = max sobre a de soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) ), para todo o s em S

No limite
    U converge para U*
```

Critério de paragem. Mede-se a maior diferença entre a utilidade nova e a anterior e para-se quando é menor ou igual a um limiar de convergência.

```
delta = max sobre s de | U_novo(s) - U_anterior(s) |
parar quando delta <= delta_max
```

### Implementação da utilidade

```
utilidade():
    U(s) = 0, para todo o s em S
    repetir
        Uant = cópia de U
        delta = 0
        para cada s em S
            U(s) = max sobre a em A(s) de Uaccao(s, a, Uant)
            delta = max(delta, |U(s) - Uant(s)|)
    enquanto delta > delta_max
    devolver U
```

Utilidade de ação

```
Uaccao(s, a, U):
    devolver soma sobre sn em suc(s) de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) )
```

### Implementação da política

```
politica(U):
    pi = {}
    para cada s em S
        se A(s) tem ações
            pi(s) = argmax sobre a em A(s) de Uaccao(s, a, U)
    devolver pi
```

Cada iteração usa a cópia da utilidade anterior. A função suc(s) devolve os estados seguintes possíveis a partir de s.

## 19. Determinismo e não determinismo

O slide distingue três combinações entre o modelo e o ambiente.

| Modelo | Ambiente |
|---|---|
| determinista | determinista |
| determinista | não determinista |
| não determinista | não determinista |

## 20. Iteração de valor, síncrona e assíncrona

A iteração de valor pode ser vista como uma propagação de informação de valor através do espaço de estados.

* **Síncrona**, as iterações atualizam sistematicamente cada estado. Mantém-se uma cópia da utilidade anterior.
* **Assíncrona**, as iterações não atualizam sistematicamente cada estado. É necessário continuar a iterar até que todos os estados tenham sido atualizados.

## 21. Limitações dos PDM

* **Dimensão dos espaços de estados**, o problema da dimensionalidade.
* **Modelos desconhecidos**, em que T e R não se conhecem e a equação de Bellman não se pode calcular.

A resolução destes problemas faz-se por aprendizagem por reforço, tratada na parte seguinte.

---

# Parte 2, Aprendizagem por Reforço

## 1. O que é

A aprendizagem por reforço resolve o problema de decisão sequencial com base na experiência, de forma iterativa, por interação direta com o ambiente.

* **Aprendizagem de comportamentos**, ou seja, de padrões de ação.
* **Aprendizagem a partir da interação com o ambiente.**
* **Aprendizagem em continuidade**, porque ocorre ao longo da operação do agente no ambiente.

Aprende-se como fazer. A aprendizagem estabelece a relação entre estados e ações, com base numa medida de desempenho, a recompensa, obtida por interação direta com o ambiente. O problema de decisão sequencial resolve-se de forma iterativa, através da ação concreta no ambiente.

```
Agente    envia ao Ambiente      a ação a
Ambiente  devolve ao Agente      o estado s e a recompensa r
```

A recompensa r é usada como reforço, e representa um ganho ou uma perda.

## 2. A pergunta

Qual a ação que um agente deve realizar num estado s, e qual é a política de ação para cada estado.

Com o princípio da decisão ótima,

```
pi*(s) = argmax sobre a em A(s) de U(s, a)

U(s, a) = E[ R(s, a, sn) + gama * U(sn) ]
```

Do estado s, cada ação a1, a2, ..., am leva a estados sn_1, ..., sn_n, com recompensas r1, ..., rn.

## 3. Utilidade de ação com modelo

Com os modelos de transição T e de recompensa R conhecidos, o valor esperado calcula-se com uma soma sobre os estados seguintes.

```
U(s, a) = E[ R(s, a, sn) + gama * U(sn) ]
U(s, a) = soma sobre sn de T(s, a, sn) * ( R(s, a, sn) + gama * U(sn) )
E(x)    = soma, para i de 0 a n, de p(xi) * xi
```

## 4. O problema, modelos desconhecidos

Quando os modelos T(s, a, sn) e R(s, a, sn) são desconhecidos não se consegue calcular a equação de Bellman nem a utilidade de ação com o modelo.

## 5. Estimativa de valor de ação por amostragem

Quando os modelos não são conhecidos, é possível estimar o valor de ação Q(s, a) por amostragem. Cada experiência dá uma amostra r de R(s, a, sn).

```
U(s, a) é aproximado por Q(s, a)
U(sn)   é aproximado por Q(sn, an), com pi(sn) = an

Q(s, a) = E[ r + gama * Q(sn, an) ]

pi*(s) é aproximado por argmax sobre a em A(s) de Q(s, a)
```

Para um conjunto finito de n amostras (s, a, r, sn, an), a estimativa calcula-se com base nas amostras observadas.

```
E(x) é aproximado por (1/n) * soma, para i de 1 a n, de xi
E(x) = limite, com n a tender para infinito, de (1/n) * soma, para i de 1 a n, de xi

Q(s, a) = (1/n) * soma, para i de 1 a n, de ( ri + gama * Q(sn, an) )
```

A aprendizagem é incremental a partir da experiência, na sequência

```
s, a, r, sn, an, ...
```

## 6. Forma incremental da estimativa

Seja J(i) = r(i) + gama * Q(sn, an) o retorno da amostra i.

```
Q(n)(s, a) = (1/n) * soma, para i de 1 a n, de J(i)

Q(n)(s, a) = alfa * ( J(n) + soma, para i de 1 a n-1, de J(i) )               com alfa = 1/n
Q(n)(s, a) = alfa * ( J(n) + (n-1) * (1/(n-1)) * soma, para i de 1 a n-1, de J(i) )
Q(n)(s, a) = alfa * ( J(n) + (n-1) * Q(n-1)(s, a) )
Q(n)(s, a) = alfa * ( J(n) + n * Q(n-1)(s, a) - Q(n-1)(s, a) )
Q(n)(s, a) = Q(n-1)(s, a) + alfa * ( J(n) - Q(n-1)(s, a) )
```

Com a substituição de J,

```
Q(n)(s, a) = Q(n-1)(s, a) + alfa * ( r(n) + gama * Q(n-1)(sn, an) - Q(n-1)(s, a) )
```

Os termos têm este significado.

| Termo | Significado |
|---|---|
| Q(n-1)(s, a) | estimativa anterior do valor de realizar a ação a no estado s |
| r(n) + gama * Q(n-1)(sn, an) | retorno esperado |
| retorno esperado menos estimativa anterior | erro de estimativa |

O erro de estimativa é a diferença entre o retorno observado, que junta a recompensa recebida com a estimativa do valor do que se segue, e a estimativa de valor da ação a.

## 7. Aprendizagem por diferença temporal

O processo atualiza a estimativa de valor de ação com base na mudança desse valor entre instantes sucessivos, a diferença temporal.

```
Nova estimativa = Estimativa + alfa * ( Retorno - Estimativa )
```

O fator de aprendizagem alfa está entre 0 e 1. O passo de atualização da estimativa de valor Q(s, a) é

```
Q(s, a) = Q(s, a) + alfa * ( r + gama * Q(sn, an) - Q(s, a) )
```

## 8. Convergência

Os valores da função Q convergem no limite se

* cada par estado-ação (s, a) for visitado um número ilimitado de vezes
* o parâmetro alfa tender para 0 no limite, com alfa(s, a) = 1/n, em que n é o número de vezes que a ação a foi selecionada no estado s

## 9. Problemas não estacionários

Num problema não estacionário a distribuição de probabilidades muda com o tempo. Nesse caso alfa pode assumir valores diferentes de 1/n, o que possibilita uma ponderação não linear das recompensas ao longo do tempo.

O fator de aprendizagem alfa, entre 0 e 1, determina a relevância das recompensas em função do tempo.

* alfa a tender para 0, maior relevância das recompensas mais antigas
* alfa a tender para 1, maior relevância das recompensas mais recentes

## 10. Exploração e aproveitamento

A aprendizagem tem duas vertentes.

**Exploração**
* Escolher uma ação que permita explorar o mundo para melhorar a aprendizagem.
* Tem por objetivo explorar todos os estados possíveis, tentando todas as ações.
* É a ação exploratória, que serve para obter experiência.

**Aproveitamento**
* Escolher a ação que leva à melhor recompensa de acordo com a aprendizagem.
* É a ação sôfrega (greedy).
* Usa o conhecimento resultante da aprendizagem para obter o máximo de recompensa.

Formalmente, com as estimativas de valor de uma ação a

```
Qt(a) é aproximado por Q*(a)
ação sôfrega    a*(t) = argmax sobre a de Qt(a)

Aproveitamento  a(t) = a*(t)
Exploração      a(t) diferente de a*(t)
```

## 11. Estratégias de seleção de ação

**Estratégia sôfrega**

```
a(t) = a*(t) = argmax sobre a de Qt(a)
```

**Estratégia epsilon-greedy.** Com epsilon a probabilidade de exploração,

```
a(t) = a*(t)          com probabilidade 1 - epsilon
a(t) = ação aleatória com probabilidade epsilon
```

É um compromisso entre exploração e aproveitamento.

**O dilema de explorar ou aproveitar.** Só aproveitar deixa o agente a repetir as ações que estimou com mais valor no início. Só explorar faz com que nunca use o que aprendeu. A questão é saber quando é que se aprendeu o suficiente para começar a aplicar o que se aprendeu.

**Exemplo de Sutton e Barto (2020).** Problema de seleção de ação com 10 ações possíveis. O valor q*(a) de cada ação é indicado, e as recompensas são geradas por distribuições de probabilidade com média q*(a) e variância 1. Os gráficos do slide mostram a recompensa média obtida e a percentagem de escolha da ação ótima em função do número de passos.

## 12. Pseudocódigo da seleção de ação epsilon-greedy

```
accao_sofrega(s):
    baralhar(accoes)                     escolha aleatória entre ações com igual valor
    devolver argmax sobre a em accoes de Q(s, a)

accao_exploracao():
    devolver escolha(accoes)

seleccionar_accao(s):
    se random() > epsilon
        accao = accao_sofrega(s)         aproveitar conhecimento
    senão
        accao = accao_exploracao()       explorar
    devolver accao
```

Como a ação aleatória é escolhida entre todas as ações, pode coincidir com a ação sôfrega.

## 13. Memória de aprendizagem

A memória de aprendizagem guarda as estimativas Q(s, a), numa função de (s, a) para os números reais.

```
iniciar(valor_omissao = 0.0):
    valor_omissao = valor_omissao
    Q = {}                      memória de aprendizagem
    estados = conjunto vazio    estados observados

q(s, a):
    devolver Q.get((s, a), valor_omissao)

actualizar(s, a, q):
    estados.add(s)
    Q[(s, a)] = q

obter_estados():
    devolver estados
```

Um par (s, a) ainda não guardado devolve o valor por omissão.

## 14. Algoritmo SARSA

O SARSA é um controlo de aprendizagem com política de seleção de ação única, ou seja, on-policy.

* Usa a mesma política para a ação e para a aprendizagem.
* Explora todas as ações, por exemplo com a estratégia epsilon-greedy.

O nome vem da sequência de cada passo de aprendizagem, State, Action, Reward, State, Action.

```
s, a, r, sn, an
```

Passos

```
1. No estado s executar a ação a, observar r e sn
2. Escolher an a partir de sn, com base numa política derivada de Q (por exemplo epsilon-greedy)
3. Atualizar Q
       Q(s, a) = Q(s, a) + alfa * ( r + gama * Q(sn, an) - Q(s, a) )
4. Atualizar s = sn e a = an
```

Passo de aprendizagem

```
aprender(s, a, r, sn, an):
    Q(s, a) = Q(s, a) + alfa * ( r + gama * Q(sn, an) - Q(s, a) )
```

Algoritmo completo

```
1. Iniciar Q(s, a)
2. Repetir, por cada episódio
3.     Iniciar s
4.     Escolher a a partir de s, com base numa política derivada de Q (por exemplo epsilon-greedy)
5.     Repetir, por cada passo
6.         Executar a ação a, observar r e sn
7.         Escolher an a partir de sn, com base numa política derivada de Q
8.         aprender(s, a, r, sn, an)
9.         Atualizar s = sn e a = an
10.    Até s ser um estado terminal
```

Como a estratégia tem exploração, a ação an escolhida em sn pode ser uma ação de exploração, e é essa que o agente executa a seguir.

## 15. Retorno em estados terminais

Nos estados terminais, que correspondem ao fim de um episódio, não existem ações seguintes a realizar. O retorno J(s, a) depende, por isso, de o passo ser de continuação ou final.

```
Continuação de episódio
    s, a, r, sn, an, ...
    J(s, a) = r + gama * Q(sn, an)         depende de recompensas futuras

Fim de episódio, sn é terminal
    s, a, r, sn
    J(s, a) = r                            só depende da recompensa observada
```

---

# Ligação entre as duas partes

| Aspeto | PDM com modelo (PDS) | Aprendizagem por reforço (APF1) |
|---|---|---|
| Conhecimento do mundo | T e R conhecidos | T e R desconhecidos |
| Valor da ação | Uaccao(s, a, U), soma sobre sn de T * (R + gama * U(sn)) | Q(s, a), estimado por amostras |
| Como se obtém | programação dinâmica, iteração de valor sobre uma cópia de U | experiência no ambiente, atualização incremental |
| Atualização | U(s) = max sobre a de Uaccao(s, a, U) | Q(s, a) = Q(s, a) + alfa * ( J - Q(s, a) ) |
| Política | pi(s) = argmax sobre a de Uaccao(s, a, U) | pi(s) = argmax sobre a de Q(s, a) |
| Exploração | não existe, o modelo permite calcular sem agir | necessária, para visitar todos os pares (s, a) |
| Estados terminais | sem sucessores, utilidade 0 | J = r, sem ação seguinte |
| Limitações | dimensão do espaço de estados e modelos desconhecidos | convergência exige visitar cada par um número ilimitado de vezes |

A aprendizagem por reforço substitui o valor esperado calculado com o modelo por uma média de amostras, e substitui o cálculo sobre todos os estados seguintes por um passo de atualização em cada transição observada. A política resultante pode comparar-se com a que o PDM calcula quando o modelo é conhecido.

## Referências

* Luís Morgado, Processos de Decisão Sequencial, ISEL-DEI
* Luís Morgado, Aprendizagem por Reforço, Parte 1, ISEL-DEI
* S. Russell, P. Norvig, Artificial Intelligence, A Modern Approach, 3rd Ed., Prentice Hall, 2010
* R. Sutton, A. Barto, Reinforcement Learning, An Introduction, 2nd Edition, MIT Press, 2020