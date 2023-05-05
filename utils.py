import numpy as np
import math

def generate_linar_array(min, max, step=1):
    return np.arange(min, max, step)

def generate_random_int_array(size, scaller = 1, offset = 0):
    return (np.random.random(size) + offset) * scaller

def generate_gaussian_noise(shape, scaller = 1):
    return np.random.randn(shape)*scaller

def zscore_normalize(X):
    """
    Normaliza todas as colunas em X
    
    Argumentos:
      X (ndarray (m,n))     : Dados de entrada, m amostras (linhas), n características (colunas)
      
    Retorna:
      X_norm (ndarray (m,n)): Matriz X normalizada
      mu (ndarray (n,))     : média de cada característica
      sigma (ndarray (n,))  : desvio padrão de cada característica
    """
    # Encontrando a média de cada característica/coluna
    mu     = np.mean(X, axis=0)                 # mu terá shape (n,) ---> axis=0 indica que a operação será feita ao longo das linhas, para cada coluna
    # Encontrando o desvio padrão de cada característica/coluna
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # O comando abaixo subtrai a média mu de cada coluna para cada exemplo, e divide pelo desvio correspondente
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
