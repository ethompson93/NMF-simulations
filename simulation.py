
import numpy as np
from numpy.random import normal
import scipy
from scipy import stats
from scipy.stats import beta

def simulation(n_seeds, n_targets, n_signals):
   
    mixing_matrix = np.random.rand(n_seeds, n_signals)
    mixing_matrix = mixing_matrix/np.sum(mixing_matrix**2, axis = 0)
    signals = np.zeros((n_signals, n_targets))
    
    #beta parameters from fit to real aptx data
    a = np.absolute(normal(0.25, 0.1, n_signals))
    b = np.absolute(normal(640, 550, n_signals))
    scale = np.absolute(normal(0.32, 0.26, n_signals))
    
    #generate signals with beta distributions
    for i in range(n_signals):
        signals[i,:] = beta.rvs(a[i], b[i], loc=0, scale=scale[i], size=n_targets)
    
    #rescale
    signals = np.exp(signals) - 1  
    
    #normalise
    signals = signals/(np.amax(signals, axis=1, keepdims=True) + 0.001)
    
    return signals, mixing_matrix
    