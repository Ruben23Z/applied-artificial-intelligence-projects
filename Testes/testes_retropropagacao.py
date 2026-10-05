from funcao_ativacao.tanh import Tanh
from rede_neuronal import RedeNeuronal

"""
Exemplo dos slides 33-36 (Parte 3): reconhecimento de barras
verticais numa imagem 3x3.
Forma da rede=[9,3,1], alpha=0.2, erro_max=0.05, épocas=1000,
ativação=tanh. (O momento não faz parte desta entrega, por isso não há beta.)

Referência (slide 35): erro final ≈ 0.049937, época ≈ 214,
saídas ≈ [0.00039, 0.96084, 0.97566, 0.95017, -0.0035, -0.00469, 0.00007]
(valores de referência dependem da inicialização aleatória dos
pesos, pelo que servem de comparação aproximada, não exata)
"""

if __name__ == "__main__":
    print("\n === Teste reconhecimento de barras verticais ===")

    entradas = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
    ]
    saidas_ref = [[0], [0.96], [0.98], [0.95], [0], [0], [0]]  # saídas da rede no slide 36, só para comparar
    saidas = [[0], [1], [1], [1], [0], [0], [0]]

    rede = RedeNeuronal([9, 3, 1], Tanh())

    alpha = 0.2
    erro_max = 0.05
    num_epocas = 1000
    beta = 0.1
    rede.treinar(entradas, saidas, num_epocas, erro_max, alpha, beta)

    print("Previsão da rede após treino (dados de treino):")
    for x, y_alvo in zip(entradas, saidas_ref):
        y = rede.propagar(x)
        print(f"x={x} -> y={y} (referência do slide ={y_alvo})")

    # Verifica classificação correta (sinal da saída face ao alvo)
    for x, y_alvo in zip(entradas, saidas):
        y = rede.propagar(x)[0]
        if y_alvo[0] == 1:
            assert y > 0.5, f"Esperava saída próxima de 1 para {x}, obtido {y}"
        else:
            assert y < 0.5, f"Esperava saída próxima de 0 para {x}, obtido {y}"

    print("Rede aprendeu a reconhecer barras verticais corretamente")
