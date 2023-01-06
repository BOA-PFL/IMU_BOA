# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:04:52 2022

Code to examine angles provided by vicon IMUs

@author: Eric.Honert
"""

# Import Libraries
import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2
import scipy
import scipy.interpolate
from scipy.integrate import cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import time
import addcopyfighandler


# Functions
def align_fuse_extract_IMU_angles(LGdat,HGdat):
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
    glo_ang : numpy array (Nx3)
        global angle from IMU

    """
    # LGdat: low-g dataframe (contains accelerometer and gyroscope)
    # HGdat: high-g data frame (contains only accelerometer)
    
    # Need to align the data
    # 1: get the data collection frequency from the low-g acclerometer
    acc_lg = np.array(LGdat.iloc[:,2:5])
    gyr = np.array(LGdat.iloc[:,5:8])
    glo_ang = np.array(LGdat.iloc[:,8:11])
    
    dum = np.array(HGdat.iloc[:,2:])
    
    HGtime = np.array(HGdat.iloc[:,0])
    LGtime = np.array(LGdat.iloc[:,0])
    
    idx = (LGtime < np.max(HGtime))*(LGtime > np.min(HGtime))
    LGtime = LGtime[idx]
    acc_lg = acc_lg[idx,:]
    gyr = gyr[idx,:]
    glo_ang = glo_ang[idx,:]
    
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
        glo_ang = glo_ang[lag:,:]
        resamp_HG = resamp_HG[:-lag,:]
        
    elif lag < 0:
        LGtime = LGtime[:lag]
        gyr = gyr[:lag,:]
        acc_lg = acc_lg[:lag,:]
        glo_ang = glo_ang[:lag,:]
        resamp_HG = resamp_HG[-lag:,:]
    
    acc = acc_lg
    
    # Find when the data is above/below 16G and replace with high-g accelerometer
    for jj in range(3):
        idx = np.abs(acc[:,jj]) > (9.81*16-0.1)
        acc[idx,jj] = resamp_HG[idx,jj]
    
    return [LGtime,acc,gyr,glo_ang]

def filtIMUsig(sig_in,cut,t):
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = np.array([sig.filtfilt(b, a, sig_in[:,jj]) for jj in range(3)]).T    
    return(sig_out)

def findEdgeAng_gyr(gyr,t,turn_idx):
    
    edgeang_dwn = []
    edgeang_up = []
    
    for jj in range(len(turn_idx)-1):
        roll_ang = cumtrapz(gyr[turn_idx[jj]:turn_idx[jj+1],2],t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
        # Remove drift
        slope = roll_ang[-1]/(len(roll_ang)-1)
        drift = (slope*np.ones([len(roll_ang)]))*(np.array(range(len(roll_ang))))
        roll_ang = roll_ang-drift
        edgeang_dwn.append(np.max(roll_ang))
        edgeang_up.append(abs(np.min(roll_ang)))
    
    return(edgeang_dwn,edgeang_up)


acc_cut = 10
gyr_cut = 6

st_idx = []
en_idx = []
trial_name = []

edgeang_dwn_gyr = []
edgeang_up_gyr = []
Side = []
TurnTime = []

# Obtain IMU signals
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\SkiValidation_Dec2022\\IMU\\'

# Load in the trial segmentation variable if it is in the directory
if os.path.exists(fPath+'TrialSeg.npy') == True:
    trial_segment_old = np.load(fPath+'TrialSeg.npy')
    trial_name = np.ndarray.tolist(trial_segment_old[0,:])
    st_idx = np.ndarray.tolist(trial_segment_old[1,:])
    en_idx = np.ndarray.tolist(trial_segment_old[2,:])
    
# High and Low G accelerometers: note that the gyro is in the low G file
L_Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03391')]
L_Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03391')]

R_Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03399')]
R_Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03399')]


for ii in range(0,4):#len(L_Lentries)):
    # Grab the .csv files
    print(L_Lentries[ii])
    LLdf = pd.read_csv(fPath + L_Lentries[ii],sep=',', header = 0)
    LHdf = pd.read_csv(fPath + L_Hentries[ii],sep=',', header = 0)
    
    RLdf = pd.read_csv(fPath + R_Lentries[ii],sep=',', header = 0)
    RHdf = pd.read_csv(fPath + R_Hentries[ii],sep=',', header = 0)
    
    # Combine the IMU data
    [Ltime,Lacc,Lgyr,Lang] = align_fuse_extract_IMU_angles(LLdf,LHdf)   
    [Rtime,Racc,Rgyr,Rang] = align_fuse_extract_IMU_angles(RLdf,RHdf)
    
    
    if (Rtime[0]-Ltime[0]) > 0:
        # The left IMU has more initial data
        idx = Ltime > Rtime[0]
        Ltime = Ltime[idx]
        Lang = Lang[idx,:]
        Lacc = Lacc[idx,:]
        Lgyr = Lgyr[idx,:]
    elif (Ltime[0]-Rtime[0]) > 0:
        # The right IMU has more initial data
        idx = Rtime > Ltime[0]
        Rtime = Rtime[idx]
        Rang = Rang[idx,:]
        Racc = Racc[idx,:]
        Rgyr = Rgyr[idx,:]
    
    # Convert the time
    Ltime = (Ltime - Ltime[0])*(1e-6)
    Rtime = (Rtime - Rtime[0])*(1e-6)
    
    Lgyr_det = filtIMUsig(Lgyr,0.5,Ltime)
    Lgyr_det = Lgyr_det[:,2]
    
    Rgyr_det = filtIMUsig(Rgyr,0.5,Ltime)
    Rgyr_det = Rgyr_det[:,2]
    
    #__________________________________________________________________________
    # Trial Segmentation
    if L_Lentries[ii] in trial_name:
        # If the trial was previously segmented, grab the appropriate start/end points
        idx = trial_name.index(L_Lentries[ii])
        tmp_st = int(st_idx[idx]); tmp_en = int(en_idx[idx]);
        print('success')
    else:
        # If new trial, us UI to segment the trial
        fig, ax = plt.subplots()
        ax.plot(Lgyr_det, label = 'Left Gyro Detection Signal')
        ax.plot(Rgyr_det, label = 'Right Gyro Detection Signal')
        fig.legend()
        print('Select start and end of analysis trial')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        tmp_st = round(pts[0,0]); tmp_en = round(pts[1,0])
        st_idx.append(tmp_st)
        en_idx.append(tmp_en)
        trial_name.append(L_Lentries[ii])
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    Ltime = Ltime[tmp_st:tmp_en]; Lang = Lang[tmp_st:tmp_en,:]
    Lacc = Lacc[tmp_st:tmp_en,:]; Lgyr = Lgyr[tmp_st:tmp_en,:]
    Lgyr_det = Lgyr_det[tmp_st:tmp_en]
    
    Rtime = Rtime[tmp_st:tmp_en]; Rang = Rang[tmp_st:tmp_en,:]
    Racc = Racc[tmp_st:tmp_en,:]; Rgyr = Rgyr[tmp_st:tmp_en,:]
    Rgyr_det = Rgyr_det[tmp_st:tmp_en]
    #__________________________________________________________________________
    # Turn segmentation: Find when the turn is happening
    Lpeaks,_ = sig.find_peaks(Lgyr_det)
    Rpeaks,_ = sig.find_peaks(-Rgyr_det)
    # Visualize the peak detection
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(Lgyr_det)
    plt.plot(Lpeaks,Lgyr_det[Lpeaks],'bs')
    plt.subplot(2,1,2)
    plt.plot(-Rgyr_det)
    plt.plot(Rpeaks,-Rgyr_det[Rpeaks],'bs')
    plt.close()
    #__________________________________________________________________________
    # Find the edge angles from the gyroscope
    Lgyr = filtIMUsig(Lgyr,gyr_cut,Ltime)
    tmp_edge_dwn,tmp_edge_up = findEdgeAng_gyr(Lgyr,Ltime,Lpeaks)
    1

# Save the trial segmentation
trial_segment = np.array([trial_name,st_idx,en_idx])
np.save(fPath+'TrialSeg.npy',trial_segment)    
    



