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
from tkinter import messagebox

# Obtain IMU signals
fPath = 'Z:\\Testing Segments\\WorkWear_Performance\\EH_Workwear_MidCutStabilityII_CPDMech_Sept23\\IMU\\'

save_on = 1
debug = 0

# High and Low G accelerometers: note that the gyro is in the low G file
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('_lowg.csv') and fName.count('SLLt')]


# Functions
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
    """
    Filter nx3 IMU signals

    Parameters
    ----------
    sig_in : numpy array (Nx3)
        Signal with x,y,z components
    cut : float
        cut-off frequency
    t : numpy array or list
        time array

    Returns
    -------
    sig_out : numpy array
        Filtered signal

    """
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = np.array([sig.filtfilt(b, a, sig_in[:,jj]) for jj in range(3)]).T    
    return(sig_out)



# 
oSubject = []
oConfig = []

stabalize_time = []
badFileList = []

# Filtering frequencies
gyr_cut = 5

# Index through the low-g files
for ii in range(6,len(Lentries)):
    print(Lentries[ii])
    # Load the trials here
    Ldf = pd.read_csv(fPath + Lentries[ii],sep=',', header = 0)
    # Save trial information
    Subject = Lentries[ii].split(sep = "-")[1]
    Config = Lentries[ii].split(sep="-")[2]
    
    # Fuse the low and high-g accelerometer data together
    igyr = np.array(Ldf.iloc[:,5:])
    iacc = np.array(Ldf.iloc[:,2:5])
    
    IMUtime = np.array(Ldf.iloc[:,0])
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)
       
    # Create the signal for identifying landings
    freq = 1/np.mean(np.diff(IMUtime))
    w = gyr_cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    landing_sig = sig.filtfilt(b, a, np.linalg.norm(igyr,axis = 1))

    # Find the landings from the filtered gyroscope data
    landings = []
    perc_peak = 0.5
    IT = 1
    while len(landings) < 6 and IT < 6:
        landings,peakheights = scipy.signal.find_peaks(landing_sig[:-10000], distance = 5000,prominence = 200, height = perc_peak*np.max(landing_sig))
        if len(landings) > 8:
            landings,peakheights = scipy.signal.find_peaks(landing_sig[:-10000], distance = 10000,prominence = 200, height = perc_peak*np.max(landing_sig))
        perc_peak = perc_peak - 0.1
        IT = IT + 1  
    
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
    good_land = []
    
    
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
                good_land.append(land)
                loc_stabalize.append(start_examine)
                land_det = 1
    
    # Debugging: Creation of dialog box for looking where foot contact are accurate
    answer = True # Defaulting to true: In case "debug" is not used
    if debug == 1:
        plt.figure(ii)
        plt.plot(IMUtime,igyr_mag)
        plt.plot(IMUtime[landings],igyr_mag[landings],'ko')
        plt.plot(IMUtime[loc_stabalize],igyr_mag[loc_stabalize],'mv')
        plt.xlabel('Time [sec]')
        answer = messagebox.askyesno("Question","Is data clean?")
        plt.close()
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(Lentries[ii])
    
    if answer == True:
        for count, land in enumerate(good_land):
            stabalize_time.append(IMUtime[loc_stabalize[count]]-IMUtime[land])
            oSubject.append(Subject)
            oConfig.append(Config)
    
    
    
    
    #__________________________________________________________________________
    # Alternative method for identifying landing stabalization time
    # Require to be below the threshold & have the 0.5 sec integral be below 5%
    loc_stabalize2 = [] # second means of determining stabilization time
    for land in landings:
        # Examine 4 seconds after the landing (about 4500 frames)
        poss_stab_tf = igyr_mag[land:land + 4500] < steady_thresh
        poss_stab_idx = np.where(poss_stab_tf == True)[0] + land
        land_det = 0
        jj = 0        
        while land_det == 0 and jj < len(poss_stab_idx):
            if sum(igyr_mag[poss_stab_idx[jj]:poss_stab_idx[jj]+round(0.5*freq)]) < round(0.5*freq)*steady_thresh:
                loc_stabalize2.append(poss_stab_idx[jj])
                land_det = 1
            else:
                jj = jj+1
    #__________________________________________________________________________

        
    

outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig),
                          'StabalizeTime':list(stabalize_time)})

if save_on == 1:
    outcomes.to_csv(fPath + 'TrailStabilize.csv',header=True)

