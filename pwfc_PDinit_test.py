import time
import numpy as np
import gc

wavefrontCube = np.random.random((180000, 2800))

def loop_test():

    tstart = time.time()
    ## Filter parameters
    nVActs = 2800   # degrees of freedom, e.g. number of DM or WFS elements
    order=30
    lag=2
    timesteps = 120000 
    
    # Get the "training set" matrix D
    end_index_D = np.shape(wavefrontCube)[0]-1-lag
    D = np.zeros((nVActs*order, timesteps), dtype=np.float32)
    for i in range(timesteps): 
        D[:,i] = np.reshape(wavefrontCube[end_index_D-i-(order-1):end_index_D+1-i,:][::-1],(order*nVActs))
        
    # Get the "future" matrix P
    end_index_P = np.shape(wavefrontCube)[0]-1
    P = np.zeros((nVActs, timesteps), dtype=np.float32)
    for i in range(timesteps):
        P[:,i] = np.reshape(wavefrontCube[end_index_P-i,:], nVActs)
    tend = time.time()
    print('loop', tend-tstart)
    del D, P
    gc.collect()
    return tend-tstart

def list_comprehension_test():

    tstart = time.time()
    ## Filter parameters
    nVActs = 2800   # degrees of freedom, e.g. number of DM or WFS elements
    order=30
    lag=2
    timesteps = 120000 
    
    # Get the "training set" matrix D
    end_index_D = np.shape(wavefrontCube)[0]-1-lag
    end_index_P = np.shape(wavefrontCube)[0]-1
    D = np.array([np.reshape(wavefrontCube[end_index_D-i-(order-1):end_index_D+1-i,:][::-1], order*nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    P = np.array([np.reshape(wavefrontCube[end_index_P-i,:], nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    tend = time.time()
    print('list comprehension', tend-tstart)
    del D, P
    gc.collect()
    return tend-tstart

def noflip_test():

    tstart = time.time()
    ## Filter parameters
    nVActs = 2800   # degrees of freedom, e.g. number of DM or WFS elements
    order=30
    lag=2
    timesteps = 120000 
    
    # Get the "training set" matrix D
    end_index_D = np.shape(wavefrontCube)[0]-1-lag
    end_index_P = np.shape(wavefrontCube)[0]-1
    D = np.array([np.reshape(wavefrontCube[end_index_D-i-(order-1):end_index_D+1-i,:], order*nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    P = np.array([np.reshape(wavefrontCube[end_index_P-i,:], nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    tend = time.time()
    print('no flip', tend-tstart)
    del D, P
    gc.collect()
    return tend-tstart

def flipud_test():

    tstart = time.time()
    ## Filter parameters
    nVActs = 2800   # degrees of freedom, e.g. number of DM or WFS elements
    order=30
    lag=2
    timesteps = 120000 
    
    # Get the "training set" matrix D
    end_index_D = np.shape(wavefrontCube)[0]-1-lag
    end_index_P = np.shape(wavefrontCube)[0]-1
    D = np.array([np.reshape(np.flipud(wavefrontCube[end_index_D-i-(order-1):end_index_D+1-i,:]), order*nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    P = np.array([np.reshape(wavefrontCube[end_index_P-i,:], nVActs) for i in range(timesteps)], dtype=np.float32).transpose()
    tend = time.time()
    print('flip up', tend-tstart)
    del D, P
    gc.collect()
    return tend-tstart

noflip, flipud = [], []
for i in range(10):
    noflip.append(noflip_test())
    flipud.append(flipud_test())
print('flipud avg', np.median(flipud))
print('no flip avg', np.median(noflip))
