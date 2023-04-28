import numpy as np
import math

def generate_linar_array(min, max, step=1):
    return np.arange(min, max, step)

def generate_random_int_array(size, scaller = 1, offset = 0):
    return (np.random.random(size) + offset) * scaller

def generate_gaussian_noise(shape, scaller = 1):
    return np.random.randn(shape)*scaller
