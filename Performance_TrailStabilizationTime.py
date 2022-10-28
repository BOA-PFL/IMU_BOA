# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:00:47 2022

@author: Eric.Honert
"""

# Import Libraries
import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
from scipy.integrate import cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import addcopyfighandler

# Functions
def align_fuse_extract_IMU(LGdat,HGdat):
    """
    Function to align, fuse, and extract all IMU information
    
    The high-g accelerometer is interpolated to the low-g accelerometer using
    the UNIX time-stamps in the innate files. The two accelerometers are then
    aligned using a cross correlation. Typically this results in a 1 frame
    phase shift

    Parameters
    ----------
    LGdat : dataframe
        low-g data frame that is extracted as raw data from Capture.U. This
        dataframe contains both the low-g accelerometer and the gyroscope.
    HGdat : dataframe
        low-g data frame that is extracted as raw data from Capture.U.

    Returns
    -------
    LGtime : numpy array (Nx1)
        time (UNIX) from the low-g file
    acc : numpy array (Nx3)
        X,Y,Z fused (between low-g and high-g) acceleration from the IMU
    gyr : numpy array (Nx3)
        X,Y,Z gyroscope from the IMU

    """
    # LGdat: low-g dataframe (contains accelerometer and gyroscope)
    # HGdat: high-g data frame (contains only accelerometer)
    
    # Need to align the data
    # 1: get the data collection frequency from the low-g acclerometer
    acc_lg = np.array(LGdat.iloc[:,2:5])
    gyr = np.array(LGdat.iloc[:,5:])
    
    dum = np.array(HGdat.iloc[:,2:])
    
    HGtime = np.array(HGdat.iloc[:,0])
    LGtime = np.array(LGdat.iloc[:,0])
    
    idx = (LGtime < np.max(HGtime))*(LGtime > np.min(HGtime))
    LGtime = LGtime[idx]
    acc_lg = acc_lg[idx,:]
    gyr = gyr[idx,:]
    
    # Create an empty array to fill with the resampled/downsampled high-g acceleration    
    resamp_HG = np.zeros([len(LGtime),3])
        
    for jj in range(3):
        f = scipy.interpolate.interp1d(HGtime,dum[:,jj])
        resamp_HG[:,jj] = f(LGtime)
        
    # Cross-correlate the y-components
    corr_arr = sig.correlate(acc_lg[:,1],resamp_HG[:,1],mode='full')
    lags = sig.correlation_lags(len(acc_lg[:,1]),len(resamp_HG[:,1]),mode='full')
    lag = lags[np.argmax(corr_arr)]
    
    if lag > 0:
        LGtime = LGtime[lag:]
        gyr = gyr[lag:,:]
        acc_lg = acc_lg[lag:,:]
        resamp_HG = resamp_HG[:-lag,:]
        
    elif lag < 0:
        LGtime = LGtime[:-lag]
        gyr = gyr[:-lag,:]
        acc_lg = acc_lg[:-lag,:]
        resamp_HG = resamp_HG[lag:,:]
    
    acc = acc_lg
    
    # Find when the data is above/below 16G and replace with high-g accelerometer
    for jj in range(3):
        idx = np.abs(acc[:,jj]) > (9.81*16-0.1)
        acc[idx,jj] = resamp_HG[idx,jj]
    
    return [LGtime,acc,gyr]

def estIMU_singleleg_landings(acc,t,HS_thresh):
    """
    Function to estimate heel-strike and mid-stance indices from the IMU

    Parameters
    ----------
    acc : numpy array (Nx3)
        X,Y,Z acceleration from the IMU
    gyr : numpy array (Nx3)
        X,Y,Z gyroscope from the IMU
    t : numpy array (Nx1)
        time (seconds)
    thresh : float/int
        threshold for detecting when heel strikes (foot contacts)

    Returns
    -------
    HS : list
        Heel-strike (foot contact events) indices
    MS : list
        Mid-stance indices

    """
    
    HS_sig = (np.gradient(np.linalg.norm(acc,axis=1),t))**2 

    window = 200
    jj = 200
    
    HS = []    
    while jj < len(HS_sig)-1000:
        if HS_sig[jj] > HS_thresh:
            # Find the maximum
            HS_idx = np.argmin(acc[jj-window:jj+window,0])
            HS.append(jj-window+HS_idx)
            jj = jj+5000 # Want to avoid any detections within ~5 seconds of each other
        jj = jj+1
    
    return HS

def filtIMUsig(sig_in,cut,t):
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = np.array([sig.filtfilt(b, a, sig_in[:,jj]) for jj in range(3)]).T    
    return(sig_out)


# Obtain IMU signals
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\FocusAnkleDualDial_Midcut_Sept2022\\IMU\\'

save_on = 0
debug = 0

# High and Low G accelerometers: note that the gyro is in the low G file
Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('_highg.csv') and fName.count('and') or fName.endswith('_highg.csv') and fName.count('SLTrail')]
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('_lowg.csv') and fName.count('and') or fName.endswith('_lowg.csv') and fName.count('SLTrail')]

# 
oSubject = []
oConfig = []

stabalize_time = []

# Filtering frequencies
gyr_cut = 5

# Index through the low-g files
for ii in range(0,len(Lentries)):
    print(Lentries[ii])
    # Load the trials here
    Ldf = pd.read_csv(fPath + Lentries[ii],sep=',', header = 0)
    Hdf = pd.read_csv(fPath + Hentries[ii],sep=',', header = 0)
    # Save trial information
    Subject = Lentries[ii].split(sep = "-")[1]
    Config = Lentries[ii].split(sep="-")[2]
    
    # Fuse the low and high-g accelerometer data together    
    [IMUtime,iacc,igyr] = align_fuse_extract_IMU(Ldf,Hdf)
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)
       
    # Create the signal for identifying landings
    freq = 1/np.mean(np.diff(IMUtime))
    w = gyr_cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    landing_sig = sig.filtfilt(b, a, np.linalg.norm(igyr,axis = 1))

    # Find the landings from the filtered gyroscope data
    landings = []
    perc_peak = 0.6
    while len(landings) < 8:
        landings,peakheights = scipy.signal.find_peaks(landing_sig[:-10000], distance = 8000,prominence = 200, height = perc_peak*np.max(landing_sig))
        perc_peak = perc_peak - 0.1
    
    igyr = filtIMUsig(igyr, gyr_cut, IMUtime)
    igyr_mag = np.linalg.norm(igyr,axis = 1)
    
    # Create the steady-state threshold
    loc_max = []
    for land in landings:
        loc_max.append(np.max(igyr_mag[land-500:land+500]))
        
    steady_thresh = 0.05*np.mean(loc_max)
        
    #__________________________________________________________________________
    # Identify stabilization time
    loc_stabalize = []  # Debugging purposes
    
    # Require 0.5 second for stabalization
    for land in landings:
        # Examine 4 seconds after the landing (about 4500 frames)
        start_examine = land
        land_det = 0
        while land_det == 0 and start_examine < land + 4500: 
            tf_steady = igyr_mag[start_examine:start_examine+round(0.5*freq)] > steady_thresh
            if sum(tf_steady) > 0:
                instable_idx = np.where(tf_steady == True)[0]
                start_examine = start_examine + instable_idx[-1] + 1
            else:
                stabalize_time.append(IMUtime[start_examine]-IMUtime[land])
                oSubject.append(Subject)
                oConfig.append(Config)
                
                loc_stabalize.append(start_examine)
                land_det = 1
    #__________________________________________________________________________
    # Alternative method for identifying landing stabalization time
    # Require to be below the threshold & have the 0.5 sec integral be below 5%
    # for land in landings:
    #     # Examine 4 seconds after the landing (about 4500 frames)
    #     poss_stab_tf = igyr_mag[land:land + 4500] < steady_thresh
    #     poss_stab_idx = np.where(poss_stab_tf == True)[0] + land
    #     land_det = 0
    #     jj = 0        
    #     while land_det == 0 and jj < len(poss_stab_idx):
    #         if sum(igyr_mag[poss_stab_idx[jj]:poss_stab_idx[jj]+round(0.5*freq)]) < round(0.5*freq)*steady_thresh:
    #             stabalize.append(poss_stab_idx[jj])
    #             land_det = 1
    #         else:
    #             jj = jj+1
    #__________________________________________________________________________
    
    if debug == 1:
        plt.figure(ii)
        plt.plot(igyr_mag)
        plt.plot(landings,igyr_mag[landings],'ko')
        plt.plot(loc_stabalize,igyr_mag[loc_stabalize],'mv')
        plt.close()
    

outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig),
                          'StabalizeTime':list(stabalize_time)})

if save_on == 1:
    outcomes.to_csv('C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\FocusAnkleDualDial_Midcut_Sept2022\\TrailStabilize.csv',header=True)

