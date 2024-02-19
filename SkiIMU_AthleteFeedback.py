# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:26:37 2023

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
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames


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
        high-g data frame that is extracted as raw data from Capture.U.

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

def findEdgeAng_gyr(gyr_roll,t,turn_idx):
    """
    Find the edging (roll) angle from the gyroscope

    Parameters
    ----------
    gyr_roll : numpy array
        z-component of the gyroscope.
    t : numpy array
        time provided by the 
    turn_idx : numpy array
        the indicies that indicate the peaks in the turn detection gyroscope
        signal.

    Returns
    -------
    edgeang_dwn : list
        maximum edge angle from each detected turn from the downhill ski
    edgeang_up : list
        maximum edge angle from each detected turn from the uphill ski
    """
    
    edgeang_dwn = []
    edgeang_up = []
    
    for jj in range(len(turn_idx)-1):
        roll_ang = cumtrapz(gyr_roll[turn_idx[jj]:turn_idx[jj+1]],t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
        # Remove drift
        slope = roll_ang[-1]/(len(roll_ang)-1)
        drift = (slope*np.ones([len(roll_ang)]))*(np.array(range(len(roll_ang))))
        roll_ang = roll_ang-drift
        edgeang_dwn.append(np.max(roll_ang))
        edgeang_up.append(abs(np.min(roll_ang)))
    
    return(edgeang_dwn,edgeang_up)


gyr_cut = 6

# Debugging variables
debug = 0
badFileList = []

# Variables for first time trial segmentation
st_idx = []
en_idx = []
trial_name = []

# Biomechanical outcomes
edgeang_dwn_gyrL = []
edgeang_dwn_gyrR = []

print('Select ALL files for a subject')
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\SkiValidation_Dec2022\\IMU\\'
filename = askopenfilenames(initialdir = fPath) # Open .csv files

# This file path will need to change!
fPath = os.path.dirname(filename[0]) + '/'

# Load in the trial segmentation variable if it is in the directory
if os.path.exists(fPath+'TrialSeg.npy') == True:
    trial_segment_old = np.load(fPath+'TrialSeg.npy')
    trial_name = np.ndarray.tolist(trial_segment_old[0,:])
    st_idx = np.ndarray.tolist(trial_segment_old[1,:])
    en_idx = np.ndarray.tolist(trial_segment_old[2,:])
    
# High and Low G accelerometers: note that the gyro is in the low G file
Hentries = [fName for fName in filename if fName.endswith('highg.csv')]
Lentries = [fName for fName in filename if fName.endswith('lowg.csv')]

for ii in range(0,len(Lentries)):
    # Grab the .csv files
    tmpFile = Lentries[ii].split(sep = "/")[-1]
    print(tmpFile)
    Ldf = pd.read_csv(Lentries[ii],sep=',', header = 0)
    Hdf = pd.read_csv(Hentries[ii],sep=',', header = 0)
    
    # Combine the IMU data
    [IMUtime,iacc,igyr,iang] = align_fuse_extract_IMU_angles(Ldf,Hdf)
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)
    
    if Lentries[ii].count('03399'):
        # For the right gyro, invert the roll
        print('Right IMU')
        igyr[:,2] = -igyr[:,2] 
    
    igyr_det = filtIMUsig(igyr,0.5,IMUtime)
    igyr_det = igyr_det[:,2]
    
    #__________________________________________________________________________
    # Trial Segmentation
    if tmpFile in trial_name:
        # If the trial was previously segmented, grab the appropriate start/end points
        idx = trial_name.index(tmpFile)
        tmp_st = int(st_idx[idx]); tmp_en = int(en_idx[idx]);
    else:
        # If new trial, us UI to segment the trial
        fig, ax = plt.subplots()
        ax.plot(igyr_det, label = 'Gyro Detection Signal')
        fig.legend()
        print('Select start and end of analysis trial')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        tmp_st = round(pts[0,0]); tmp_en = round(pts[1,0])
        st_idx.append(tmp_st)
        en_idx.append(tmp_en)
        trial_name.append(tmpFile)
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    IMUtime = IMUtime[tmp_st:tmp_en]; iang = iang[tmp_st:tmp_en,:]
    iacc = iacc[tmp_st:tmp_en,:]; igyr = igyr[tmp_st:tmp_en,:]
    igyr_det = igyr_det[tmp_st:tmp_en]
    
    #__________________________________________________________________________
    # Turn segmentation: Find when the turn is happening
    ipeaks,_ = sig.find_peaks(igyr_det, height = 10, distance = 200)
    # Visualize the peak detection
    answer = True # Defaulting to true: In case "debug" is not used
    if debug == 1:
        plt.figure()
        plt.plot(igyr_det)
        plt.plot(ipeaks,igyr_det[ipeaks],'bs')
        answer = messagebox.askyesno("Question","Is data clean?")
        plt.close()
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(Lentries[ii])
    
    if answer == True:
        print('Estimating point estimates')
        #______________________________________________________________________
        # Find the edge angles from the gyroscope
        igyr = filtIMUsig(igyr,gyr_cut,IMUtime)
        tmp_edge_dwn,_ = findEdgeAng_gyr(igyr[:,2],IMUtime,ipeaks)
        if Lentries[ii].count('03399'):
            edgeang_dwn_gyrR.extend(tmp_edge_dwn)
        else:
            edgeang_dwn_gyrL.extend(tmp_edge_dwn)

# Save the trial segmentation
trial_segment = np.array([trial_name,st_idx,en_idx])
np.save(fPath+'TrialSeg.npy',trial_segment)

# Feedback for the athletes
L_avgedge = np.mean(edgeang_dwn_gyrL)
R_avgedge = np.mean(edgeang_dwn_gyrR)
avgDiff = abs(R_avgedge - L_avgedge)

print('Average Edge Angle: ', round(np.mean(edgeang_dwn_gyrL + edgeang_dwn_gyrR)))
if R_avgedge > L_avgedge:
    print("Skier has ", round(avgDiff), "greater edge angle on right foot")

if L_avgedge > R_avgedge:
    print("Skier has ", round(avgDiff), "greater edge angle on left foot")


