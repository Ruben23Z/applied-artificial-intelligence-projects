# Inteligência Artificial para Sistemas Cognitivos

## 1. Introdução à Inteligência Artificial

**Definição**: área científica e tecnológica que estuda o desenvolvimento de sistemas computacionais capazes de comportamento inteligente, automatizando e amplificando capacidades cognitivas humanas.

### Tipos de IA
- **IA Restrita (Fraca)** — resolve uma tarefa específica (ex.: reconhecimento de imagem).
- **IA Geral (Forte)** — capacidade de aplicar conhecimento em contextos diversos, semelhante ao ser humano (ainda teórica).

### Paradigmas principais
| Paradigma | Ideia central |
|---|---|
| Simbólico | Inteligência resulta de processos computacionais sobre estruturas simbólicas |
| Conexionista | Inteligência emerge da interação de unidades elementares interligadas (redes neuronais) |

### Inteligência, Cognição e Racionalidade
- **Cognição** — representação e processamento de conhecimento (modelação, aquisição, inferência).
- **Racionalidade** — capacidade de agir para obter o melhor resultado possível dado o conhecimento disponível [Russell & Norvig].
  - **Ilimitada** → otimização (melhor comportamento possível).
  - **Limitada** → satisfação (comportamento suficientemente bom dados os recursos disponíveis).
- **Autonomia** e **Flexibilidade** são as duas expressões principais da capacidade cognitiva de um sistema.

### Desafios da IA
- Ética (decisão, responsabilidade)
- Privacidade e segurança
- Transparência e explicabilidade (opacidade de modelos como redes neuronais profundas)

---

## 2. Redes Neuronais Artificiais

### Aprendizagem Automática
- **Aprendizagem** = melhoria de desempenho (D) numa tarefa (T) com a experiência (E).
- Aprendizagem ≠ memorização → requer **generalização**.
- **Abordagens**: supervisionada, não supervisionada, por reforço.

### Modelo do Neurónio Artificial
Componentes: entradas `x`, pesos `w`, função de transferência `f`, saída `y`.

```
h = w·x + b        (soma de ativação)
y = φ(h)           (função de ativação)
```

- `b` (pendor/bias) desloca a fronteira de discriminação.
- Geometricamente, um neurónio define um hiperplano de separação no espaço de entradas.

### Funções de Ativação
| Função | Intervalo | Derivável |
|---|---|---|
| Degrau (Step) | {0,1} | Não |
| Sigmoide | [0,1] | Sim: `φ'(h) = φ(h)(1-φ(h))` |
| Tangente hiperbólica (tanh) | [-1,1] | Sim: `φ'(h) = 1 - tanh²(h)` |
| ReLU | [0,∞) | Diferencial exceto em `h = 0` |

A não linearidade é essencial para discriminar padrões e para o **Teorema da Aproximação Universal**.

### Problema XOR e Redes Multicamada
- Um único neurónio (perceptrão) só resolve problemas **linearmente separáveis** (AND, OR).
- **XOR não é linearmente separável** → requer pelo menos uma camada escondida.
- Redes multicamada realizam a separação não linear transformando o espaço de entrada em espaços de dimensão superior.

### Teorema da Aproximação Universal
Uma rede neuronal com **uma única camada escondida**, com número finito de neurónios e função de ativação não linear, pode aproximar qualquer função contínua num intervalo compacto com precisão arbitrária.
> O teorema garante *existência*, não indica *arquitetura nem pesos concretos*.

### Representação Formal
Uma rede é uma função `f: ℝⁿ → ℝᵏ`, com parâmetros `θ = {W², b², ..., Wᴺ, bᴺ}`.

```
yⁿ = φⁿ(hⁿ),   hⁿ = xⁿWⁿ + bⁿ,   xⁿ = yⁿ⁻¹
```

### Algoritmo de Retropropagação (Backpropagation)
Formulado como problema de **otimização** por descida de gradiente:

```
θ ← θ − α∇L(θ)
```

- `α` = taxa de aprendizagem
- Regra da cadeia usada para propagar o erro da saída até às camadas escondidas
- `δⱼⁿ` = componente de propagação do erro (derivada do erro em relação à saída da camada)

Para a camada de saída (erro quadrático médio): `δₖᴺ = yₖᴺ − yₖ`

**Etapas do algoritmo**:
1. Propagação direta (calcular saída da rede)
2. Cálculo do erro (função de perda)
3. Retropropagação do erro (da saída para a entrada)
4. Atualização dos pesos e pendores


### Descida de Gradiente — Variantes
| Método | Ideia |
|---|---|
| Descida de Gradiente (DG) | Atualização com base no gradiente total |
| Descida de Gradiente com Momento | Adiciona memória de variações anteriores; pode ajudar a atravessar regiões de gradiente baixo/nulo e a melhorar a convergência, mas **não elimina nem garante resolver** o problema dos mínimos locais |
| Descida de Gradiente Estocástica (SGD) | Usa subconjuntos (*batches*) aleatórios — mais rápido, mais ruidoso |

**Problemas comuns**: mínimos locais, convergência lenta (passo pequeno), oscilação (passo grande).

### Processo de Aprendizagem
1. Pré-processamento (limpeza, normalização, codificação, separação treino/teste)
2. Extração de características
3. Treino (arquitetura, função de perda, otimizador, épocas)
4. Avaliação e ajuste

### Arquitetura de Rede
- **Nº de neurónios de entrada** = nº de características.
- **Nº de neurónios de saída**: 1 (classificação binária/regressão) ou nº de classes (multi-classe).
- **Nº de neurónios/camadas escondidas**: não existe fórmula exacta — a definição da arquitetura tem uma componente **experimental**. Uma *heurística* usada como ponto de partida é `(nº entradas + nº saídas) / 2`, mas o desempenho de diferentes configurações deve ser verificado progressivamente.

### Sobreparametrização (Overfitting)
- **Causas**: treino excessivo, poucos dados, ruído, demasiados parâmetros.
- **Soluções**: simplificação, *early stopping*, *data augmentation*, regularização, *dropout*.

### Redes Convolucionais (CNN)
- Camadas: **convolucionais** (extração de padrões), **pooling** (redução da dimensionalidade), **densas** (classificação final).
- Vantagens: invariância à translação, redução de parâmetros, extração automática de características.

---

## 3. Aprendizagem por Reforço

Resolve problemas de **decisão sequencial** por interação direta com o ambiente: `Agente ⇄ (ação a, estado s, recompensa r) ⇄ Ambiente`.

### Conceitos-chave
- **Utilidade de ação** `U(s,a) = E[R(s,a,s') + γU(s')]`
- **Política ótima**: `π*(s) = argmax_a U(s,a)`
- **Estimativa de valor Q(s,a)** — usada quando os modelos de transição/recompensa são desconhecidos.

### Algoritmos fundamentais
| Algoritmo | Tipo | Regra |
|---|---|---|
| SARSA | On-policy | `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]` |
| Q-Learning | Off-policy | `Q(s,a) ← Q(s,a) + α[r + γ·maxₐ'Q(s',a') − Q(s,a)]` |

### Exploração vs. Aproveitamento
| Estratégia | Ideia |
|---|---|
| ε-greedy | Explora com probabilidade ε fixa |
| ε-greedy com decaimento | ε diminui ao longo do tempo |
| Valores iniciais otimistas | Promove exploração inicial |
| UCB (Upper Confidence Bounds) | Considera incerteza no valor da ação |
| Softmax | Probabilidade proporcional ao valor estimado (parâmetro *temperatura*) |

### Aceleração da aprendizagem
- **Eligibility traces** (SARSA(λ))
- **Aprendizagem com modelos** (Model-Based RL, arquitectura Dyna-Q)
- **Memória de experiência** (*Experience Replay*)

### Aprendizagem por Reforço Profunda (Deep RL)
- Usa redes neuronais para aproximar `Q(s,a)` quando o espaço de estados é muito grande (ex.: imagens de jogos Atari).
- **Deep Q-Learning** [Mnih et al., 2015]: usa **memória de experiência** + **duas redes** (previsão e alvo) para estabilizar o treino.

---

## 4. Processos de Decisão Sequencial (MDP)

Um **Processo de Decisão de Markov** é definido por: `(S, A, T, R, γ)`
- `S` — estados; `A(s)` — ações possíveis; `T(s,a,s')` — probabilidade de transição; `R(s,a,s')` — recompensa; `γ` — factor de desconto.

### Propriedade de Markov
O estado futuro só depende do estado presente (não do histórico completo).

### Utilidade e Retorno
```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...
```
- **Equação de Bellman**: `U*(s) = max_a Σ T(s,a,s')[R(s,a,s') + γU*(s')]`
- Resolvido por **Programação Dinâmica** (iteração de valor).

---

## 5. Raciocínio Automático

Dois tipos de problemas:
- **Planeamento** — sequência de ações/situações do estado inicial ao objetivo.
- **Otimização** — melhor configuração de parâmetros.

### Procura em Espaços de Estados
- **Estado**, **transição**, **espaço de estados** (grafo).
- **Métodos não informados**: profundidade (DFS), largura (BFS), custo uniforme.
- **Métodos informados**: usam heurística `h(n)`.

| Método | f(n) | Característica |
|---|---|---|
| Custo uniforme | `g(n)` | ptimo, não usa heurística |
| Sôfrega (Greedy) | `h(n)` | Rápido, pode ser sub-ótimo |
| A* | `g(n) + h(n)` | Ótimo se `h` for admissível |

**Heurística admissível**: `0 ≤ h(n) ≤ h*(n)` (nunca sobrestima o custo real).

### Complexidade
`b` = factor de ramificação, `d` = profundidade da solução → complexidade **exponencial** `O(b^d)`.

---

## 6. Raciocínio para Otimização

### Métodos de Procura Local
- **Hill-Climbing** — substitui estado actual pelo melhor sucessor. Problema: **ótimos locais**.
  - Variantes: estocástico, único sucessor, **reinício aleatório**.
- **Simulated Annealing** — aceita soluções piores com probabilidade dependente da "temperatura" (permite escapar de ótimos locais).
- **Algoritmos Genéticos** — população de soluções, operadores de **seleção**, **cruzamento** e **mutação**; conceito de **esquema** (Teorema dos Esquemas de Holland).

---

## 7. Racionalidade Limitada

Face à complexidade computacional (recursos limitados), usam-se métodos aproximados:
- **Aproximação** (heurísticas)
- **Abstração** (redução de detalhe — grelhas hierárquicas, quadtrees, planeamento hierárquico)
- **Modularização** (decomposição em sub-problemas)
- **Focagem** (limitar espaço/tempo de procura — IDA*, RTAA*, algoritmos *anytime*)

**A\* ponderado**: `f(n) = g(n) + (1+ε)h(n)` — troca ótimo por eficiência.

---

## 8. Representação de Conhecimento

- **Dados → Informação → Conhecimento** (via contexto e inferência).
- **Redes Semânticas** — grafos de conceitos e relações (ex.: herança "é-um").
- **Grafos de Conhecimento** — triplos `(sujeito, predicado, objeto)`.
- **Lógica Proposicional e de Predicados (1ª ordem)** — predicados, quantificadores (∀, ∃), inferência (*modus ponens*, resolução).
- **Prolog** — linguagem declarativa baseada em lógica: factos, regras, unificação, *backtracking*.

---

## 9. Arquiteturas de Agentes Inteligentes

| Arquitetura | Características |
|---|---|
| **Reativa** | Resposta rápida (estímulo→resposta), sem deliberação; robusta mas limitada |
| **Deliberativa** | Raciocínio com objetivos explícitos; modelo **BDI** (Belief-Desire-Intention) |
| **Híbrida** | Combina camadas reativa (execução), tática e deliberativa (estratégica) |

### Modelo BDI
- **Crenças** (o que se sabe), **Desejos** (objetivos), **Intenções** (compromissos de execução).
- **Reconsideração**: equilíbrio entre estratégia cautelosa (reconsiderar sempre) e arrojada (nunca reconsiderar).

---

## 10. Raciocínio Social e Sistemas Multi-Agente

- **Teoria da Dependência Social** [Conte & Castelfranchi]: relações de **poder** e **dependência** entre agentes explicam a cooperação.
  - Dependência **mútua** → favorece cooperação.
  - Dependência **recíproca** → favorece troca social.
- **Comunicação entre agentes**: Atos de Discurso (Speech Acts), linguagem **KQML** (nível de conteúdo, mensagem, comunicação).
- **Coordenação**: planeamento multi-agente, estruturas organizacionais, **Protocolo Contract-Net** (gestor/contratado).

---

## Referências principais

- S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, Prentice Hall / Pearson.
- R. Sutton, A. Barto, *Reinforcement Learning: An Introduction*, MIT Press.
- Y. LeCun, Y. Bengio, G. Hinton, *Deep Learning*, Nature, 2015.
- V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.
- M. Wooldridge, *An Introduction to Multi-Agent Systems*, John Wiley & Sons.
- I. Bratko, *Prolog Programming for Artificial Intelligence*, Addison-Wesley.
