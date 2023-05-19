""" 
nossa_biblioteca_de_funcoes.py
    Trata-se de uma biblioteca onde estamos colocando todas as funções que nós achamos que podem ser úteis
"""

import math
import numpy as np
#import matplotlib.pyplot as plt


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
