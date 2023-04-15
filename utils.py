import numpy as np

def calcula_saida_do_modelo(x, w, b):
    """
    Calcula a previsão para um modelo na forma de reta
    Argumentos da função:
      x (ndarray (m,)): Conjunto de dados com m amostras 
      w,b (escalar)   : Parâmetros do modelo  
    Retorna
      y (ndarray (m,)): Previsão de saída
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m): # estrutura 'for' em Python
        f_wb[i] = w * x[i] + b
        
    return f_wb

def calcula_custo(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float: 
    """
    Calcula a função custo no âmbito da regressão linear.
    Argumentos da função:
      x (ndarray (m,)): Conjunto de dados com m amostras 
      y (ndarray (m,)): Valores alvo de saída
      w,b (escalar)   : Parâmetros do modelo  
    Retorna
      custo_total (float): O custo custo de se usar w,b como parâmetros na regressão linear
               para ajustar os dados
    """
    # número de amostras de treinamento
    m = x.shape[0] 
    
    soma_custo = 0 
    for i in range(m): 
        f_wb = w * x[i] + b  # Monta equação estimativa
        custo = (f_wb - y[i]) ** 2  # Calcula o erro quadrático
        soma_custo = soma_custo + custo  # Somatório o erro quadrático
    custo_total = (1 / (2 * m)) * soma_custo # Calcula erro quadrático médio

    return custo_total

def calcula_gradiente(x, y, w, b): 
    """
    Calcula o gradiente para Regressão Linear 
    Argumentos da função:
      x (ndarray (m,)): Conjunto de dados com m amostras 
      y (ndarray (m,)): Valores alvo de saída
      w,b (scalar)    : parâmetros do modelo
    Retorna
      dj_dw (scalar): O gradiente do custo em relação ao parâmetros w
      dj_db (scalar): O gradiente do custo em relação ao parâmetros b   
     """
    
    # Número de amostras de treinamento
    m = x.shape[0]

    # Inicializa variáveis
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        # Calcula valor estimado para determinada amostra
        f_wb = w * x[i] + b 
        # Calcula a parcial de J em w
        dj_dw_i = (f_wb - y[i]) * x[i] 
        # Calcula a parcial de J em b
        dj_db_i = f_wb - y[i] 
        # Calcula o somatório das parciais de J em w
        dj_db += dj_db_i
        # Calcula o somatório das parciais de J em b
        dj_dw += dj_dw_i
    # Calcula o gradiente de J em w
    dj_dw = dj_dw / m 
    # Calcula o gradiente de J em b
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def metodo_do_gradiente(x, y, w_in, b_in, alpha, num_iters, calcula_custo, calcula_gradiente): 
    """
    Aplica o Método do Gradiente para ajustar w,b. Atualiza w,b ao longo de 
    num_iters passos (iterações) assumindo uma taxa de aprendizado alpha
    
    Argumentos da função:
      x (ndarray (m,))  : Conjunto de dados com m amostras 
      y (ndarray (m,))  : Valores alvo de saída
      w_in,b_in (scalar): valores iniciais para os parâmetros w,b 
      alpha (float):      taxa de aprendizado
      num_iters (int):    número de iterações para o método
      calcula_custo:      função responsável por calcular o custo
      calcula_gradiente:  função responsável por calcular o gradiente
      
    Retorna:
      w (scalar): Valor atualizado para w após rodar o Método do Gradiente
      b (scalar): Valor atualizado para b após rodar o Método do Gradiente
      J_history (List): Contém o histórico dos valores de custo
      p_history (list): Contém o histórico dos valores para [w,b] 
      """
    
    # Arrays que armazenam os valores históricos de J e w para cada iteração para que seja possível fazer gráfico depois
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calcula o gradiente usando a função calcula_gradiente
        dj_dw, dj_db = calcula_gradiente(x, y, w , b)

        # Atualiza os parâmetros w,b a partir do gradiente calculado
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Salva o custo J para cada iteração
        if i < 100000:     
            J_history.append( calcula_custo(x, y, w , b))
            p_history.append([w,b])
        # Faz print em tempo real enquanto o Método do Gradiente estiver rodando
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteração {i:4}: Custo {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #retorna w,b e valores históricos


