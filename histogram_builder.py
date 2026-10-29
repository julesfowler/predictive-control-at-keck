
import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time

#from keckrtc.ext_interface.trsClient import TRS_CLIENT

def convertUT2timestamp(time_str):
    '''
    time_str (str): format of "YYYY-MM-DDTHH:MM:SS"
    taken as-is from Avinash 
    '''
    t = Time(time_str, scale='tai', format='isot')
    t0 = Time('1970-01-01T00:00:00', scale='tai', format='isot')
    dt = t - t0
    dt_ns = dt.sec * 10**9
    return dt_ns

def pull_telemetry_by_time_chunk(startUTC, length):

    client = TRS_CLIENT('10.136.1.155')
    client.connect()
    
    dm = client.getRecords('xinetics',ln=n_frames)
    wfs_resid = [info.residualWaveFront for info in dm[1]]
    
    return np.array(wfs_resid)


def calculate_rms(population):

    return np.sqrt(np.mean(population**2, axis=0))


def collect_data(start_times, lengths):
    
    wfs_resid_rms = []
    for index, time in enumerate(start_times):
        #wfs_resid = pull_telemetry_by_time_chunk(time, lengths[index])
        wfs_resid = np.random.random((lengths[index], 249))
        rms = calculate_rms(wfs_resid)
        wfs_resid_rms.append(rms)
    wfs_resid_rms = np.ravel(wfs_resid_rms)
    
    return wfs_resid_rms

def build_histograms(data, names):
    
    for index, predictor_population in enumerate(names):
        stats = plt.hist(data[index], label=predictor_population)
        mode_index = np.where(stats[0] == np.max(stats[0]))[0][0]
        mode = stats[1][mode_index]
        print(f'{predictor_population}: {mode}')

    plt.savefig('predictors_histogram.png')


## -- RUN

if __name__ == "__main__":

    # 300, 10, 60000
    fxxx_starts = ['2025-12-04T0:40:00', '2025-12-04T0:45:00', '2025-12-04T0:50:00']
    fxxx_lengths = [60000, 60000, 60000]
    fxxx_data = collect_data(fxxx_starts, fxxx_lengths)

    # 300, 10, 60000
    f250xx_starts = ['2025-12-04T0:42:00', '2025-12-04T0:47:00', '2025-12-04T0:52:00']
    f250xx_lengths = [60000, 60000, 60000]
    f250xx_data = collect_data(f250xx_starts, f250xx_lengths)

    build_histograms([f250xx_data, fxxx_data], ['alpha=250, n=10, l=60000', 'alpha=300, n=10, l=60000'])
    




