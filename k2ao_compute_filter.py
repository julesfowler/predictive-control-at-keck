#!/usr/bin/env python3

import os
#os.nice(10)

import datetime
import getopt, sys
from threading import Thread
import time

import astropy.io.fits as pf
import numpy as np
import scipy.linalg as scila

#from keckrtc.ext_interface.trsClient import TRS_CLIENT


def computePCFilter(nbSets, order, lag, n_actuators, wavefrontCube, alpha, filter_name='./newfilter.npy'):
    tstart = time.time()
    ## Filter parameters
    nVActs = n_actuators     # degrees of freedom, e.g. number of DM or WFS elements

    # Make the predictive filter
    print('nbSets = '+ str(nbSets))
    timesteps = nbSets-order-lag # total number of time steps in filter
    
    
    # Get the "training set" matrix D and "future" matrix P
    end_index_D = np.shape(wavefrontCube)[0]-1-lag
    end_index_P = np.shape(wavefrontCube)[0]-1
    D = np.array([np.reshape(wavefrontCube[end_index_D-i-(order-1):end_index_D+1-i,:][::-1], (order*nVActs)) for i in range(timesteps)], dtype=np.float64).transpose()
    #np.save('D.npy', D) 
    P = np.asarray([np.reshape(wavefrontCube[end_index_P-i,:], nVActs) for i in range(timesteps)], dtype=np.float64).transpose()
    #np.save('P.npy', P) 
       
    
    # Check that D is "lag" elements behind P
    print('Check that D is "lag" elements behind P')
    print('1st element in D = '+ str(D[0:2,0]))
    print('lag element in P = '+ str(P[0:2,lag]))
    
    t0=time.time()

    # use least squares
    # this is faster and equivalent
    alpha = float(alpha) 
    thefilter = np.array(P@D.T@scila.inv(D@D.T + alpha*np.identity(order*nVActs)), dtype='float64')
    
    t1=time.time()
    #print('time for least-squares computation ' + str(t1-t0))
    #print('total time ' + str(tstart-t1))
    
    # now see how well they agree
    diff0 = P - D[0:349,:]
    #diff1 = P - thefilter@D
    diff2 = P - thefilter@D
    print('error without prediction ' + str(np.sqrt(np.mean(diff0**2))))
    print('error with least-squares prediction ' + str(np.sqrt(np.mean(diff2**2))))
    
    # Save the filter
    np.save(filter_name, thefilter)
    print(f"saved in {filter_name}")
    
    print('FILTER SHAPE = '+str(np.shape(thefilter)))
    return thefilter


def pull_pol_telemetry_from_trs(n_frames):
    
    client = TRS_CLIENT('10.136.1.155')
    client.connect()
    
    # I absolutely just made up these keys 
    pol = client.getRecords('pwfcPOL', ln=n_frames)
    pol_ts = [info.timestamp for info in pol[1]]
    pol_slopes = [info.POLSlopes for info in pol[1]]
    
    data_start_time = pol_ts[0]
    pol_slopes = np.array(pol_slopes)

    return pol_slopes, data_start_time


def build_pol_telemetry_from_trs(n_frames, lag, reconstructor_path='/home/aodev/jfowler/PyAO/aodata/shwfs_files/reconstructor/304B.mr'):
    
    client = TRS_CLIENT('10.136.1.155')
    client.connect()
    
    wfs = client.getRecords('ocam2k', ln=n_frames+10)
    wfs_time = [info.timestamp for info in wfs[1]]
    wfs_slopes = [info.SHWFS1Centroid for info in wfs[1]]
    
    dm_start = wfs_time[lag]
    dm = client.getRecords('xinetics', start=dm_start, ln=n_frames+10, orderby='ts')
    dm_time = [info.timestamp for info in dm[1]]
    dm_cmds = [info.DMCommand for info in dm[1]]

    if dm_time[0] != wfs_time[lag]:
        print(dm_time[0], wfs_time[lag])
        raise ValueError('DM commands and WFS slopes are not appropriately lagged.')
    
    # Rip off the last 3 modes which are p/t/t
    reconstructor_matrix = np.fromfile(reconstructor_path, dtype='>f').reshape(352,608)[:-3]
    
    wfs_slopes_dm_space = np.array([np.matmul(reconstructor_matrix, slopes) for slopes in wfs_slopes])
    pol_slopes = np.array(dm_cmds)[:n_frames] + wfs_slopes_dm_space[:n_frames]
    #pol_slopes = -1*np.array(dm_cmds)[:n_frames] + wfs_slopes_dm_space[:n_frames]
    
    data_start_time = wfs_time[0] 
    
    return pol_slopes, data_start_time

def get_pol_telemetry_from_trs(n_frames, lag):
    
    client = TRS_CLIENT('10.136.1.155')
    client.connect()
    
    dm = client.getRecords('xinetics',ln=n_frames)
    pol_slopes = [info.POLCOutput for info in dm[1]]
    #print('pol slopes:', pol_slopes) 

    return np.array(pol_slopes)

if __name__ == '__main__':
    print('Predictive Control Filter Manager Tool')
    
    alpha = int(sys.argv[1])
    order = int(sys.argv[2])
    nbFrame = int(sys.argv[3])
    filter_name = str(sys.argv[4])
    lag = 1
    #order = 10 # this is history vector length / n 
    #nbFrame = 60000 #int(sys.argv[1]) # this is timesteps of training data / m
    n_actuators = 349 # 349
    
    t0=time.time()
    print("------New filter computation---------")
    print("alpha = "+str(alpha))
    print("lag = "+str(lag))
    print("order = "+str(order))
    print("nbFrame = "+str(nbFrame))
    
    #pol_data = get_pol_telemetry_from_trs(nbFrame, 1)
    pol_data = np.zeros((nbFrame, 349))
    #pol_data = np.load('MultilayerTurb_volts_r16_KAON303_TTsub.npy')[:60000] # test turb data
    #pol_data, data_start_time = build_pol_telemetry_from_trs(nbFrame, 1)
    # pol_data, data_start_time = pull_pol_telemetry_from_trs(nbFrame)
    t1 = time.time()
    #print('pol data:', pol_data)
    thefilter = computePCFilter(nbFrame, order, lag, n_actuators, pol_data, alpha, filter_name=filter_name)
    t2=time.time()

    processTime = t2-t1
    dataTime = t1 - t0
    totalTime = t2 - t0
    print("predictiveFilterMgr: total time = %.3f, data time = %.3f, processing time = %.3f\t\t\t" % (totalTime, dataTime, processTime))

