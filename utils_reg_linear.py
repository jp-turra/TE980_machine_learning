import numpy as np
import math

# Funções referentes a REGRESSÃO LINEAR

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

def calcula_gradiente_vetorizado(X, y, w, b): 
    """
    Calcula Gradiente para Regressão Linear
    Args:
      X (ndarray (m,n)): Dados, contendo m exemplos com n características
      y (ndarray (m,)) : valores alvo
      w (ndarray (n,)) : parâmetros w do modelo  
      b (scalar)       : parâmetro b do modelo
      
    Retorna:
      dj_dw (ndarray (n,)): O gradiente da função custo com relação aos parâmetros w. 
      dj_db (escalar):      O gradiente da função custo com relação ao parâmetro b. 
    """
    m,n = X.shape           #(número de exemplos, número de características)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        erro = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + erro * X[i, j]    
        dj_db = dj_db + erro                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

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

def metodo_do_gradiente_vetorizado(X, y, w_in, b_in, calcula_custo, calcula_gradiente, alpha, num_iters): 
    """
    Performa Método do Gradiente para aprender theta. Atualiza theta ao longo de  
    num_iters passos de iteração usando uma taxa de aprendizado alpha
    
    Args:
      X (ndarray (m,n))   : Dados, contendo m exemplos com n características
      y (ndarray (m,))    : valores alvo
      w_in (ndarray (n,)) : valores iniciais dos parâmetros w do modelo  
      b_in (escalar)      : valor inicial do parâmetro b do modelo
      calcula_custo       : função que calcula o custo
      calcula_gradiente   : função que calcula o gradiente
      alpha (float)       : taxa de aprendizado
      num_iters (int)     : Número de iterações para o método do gradiente
      
    Retorna:
      w (ndarray (n,)) : Valores atualizados para os parâmetros w
      b (scalar)       : Valores atualizado para o parâmetro b 
      """
    
    # Valores históricos
    J_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):

        # Calcula o gradiente
        dj_db,dj_dw = calcula_gradiente(X, y, w, b)   ##None

        # Atualiza os parâmetros
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Salva o custo
        if i<100000:      # prevent resource exhaustion 
            J_history.append( calcula_custo(X, y, w, b))

        # Faz print de tempos em tempos
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4d}: Custo {J_history[-1]:8.2f}   ")
        
    return w, b, J_history # retorna valores finais e históricos

def sigmoid(z):
    """
    Calcula o valor sigmoide de z

    Argumento:
        z (ndarray): Um escalar ou numpy array de qualquer tamanho.

    Retorna:
        g (ndarray): sigmoid(z), com o mesmo shape de z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g