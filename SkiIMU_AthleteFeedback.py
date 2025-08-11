# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:26:37 2023

@author: Eric.Honert
"""

# Import Libraries
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import addcopyfighandler
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames

print('Select ALL files for a subject')
fPath = 'Z:\\Testing Segments\\Snow Performance\\2025_Mech_AlpineRace\\IMU\\'
filename = askopenfilenames(initialdir = fPath) # Open .csv files

gyr_cut = 6
# Debugging variables
debug = 0

RIMUno = '04580'

# Functions
def delimitTrialIMU(SegSig):
    """
    Function to crop the data

    Parameters
    ----------
    SegSig : dataframe
        The signal that will allow for segmentation

    Returns
    -------
    segidx: array
        Segmentation indices

    """
    print('Select 2 points: the start and end of the trial')
    fig, ax = plt.subplots()
    ax.plot(SegSig, label = 'Segmenting Signal')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=40))
    plt.close()
    segidx = pts[:,0]
    return(segidx)

def filtIMUsig(sig_in,cut,t):
    """
    Filter 3 axes of an IMU signal (X,Y,Z) with a 2nd order butterworth filter 
    at the specified cut-off frequency

    Parameters
    ----------
    sig_in : numpy array (Nx3)
        acceleration or gyroscope signal
    cut : float
        cut-off frequency
    t : numpy array
        time (sec)

    Returns
    -------
    sig_out : numpy array (Nx3)
        filtered signal at the specified cut-off frequency

    """
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


badFileList = []

# Variables for first time trial segmentation
st_idx = []
en_idx = []
trial_name = []
# Biomechanical outcomes
edgeang_dwn_gyrL = []
edgeang_dwn_gyrR = []

# This file path will need to change!
fPath = os.path.dirname(filename[0]) + '/'
    
# High and Low G accelerometers: note that the gyro is in the low G file
Lentries = [fName for fName in filename if fName.endswith('lowg.csv')]

for ii in range(0,len(Lentries)):
    # Grab the .csv files
    tmpFile = Lentries[ii].split(sep = "/")[-1]
    print(tmpFile)
    Ldf = pd.read_csv(Lentries[ii],sep=',', header = 0)
    iacc = np.array(Ldf.iloc[:,2:5])
    igyr = np.array(Ldf.iloc[:,5:8])
    IMUtime = np.array(Ldf.iloc[:,0])
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)
    
    if Lentries[ii].count(RIMUno):
        # For the right gyro, invert the roll
        print('Right IMU')
        igyr[:,2] = -igyr[:,2] 
    
    igyr_det = filtIMUsig(igyr,0.5,IMUtime)
    igyr_det = igyr_det[:,2]
        
    #__________________________________________________________________________
    # Trial Segmentation
    if os.path.exists(Lentries[ii]+'TrialSeg.npy'):
        # Load the trial segmentation
        trial_segment = np.load(Lentries[ii]+'TrialSeg.npy', allow_pickle =True)
    else:
        # Segment the trial based on the gyroscope deteciton signal
        trial_segment = delimitTrialIMU(igyr_det)
        # Save the trial segmentation
        np.save(Lentries[ii]+'TrialSeg.npy',trial_segment)
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    TS = int(trial_segment[0]); TE = int(trial_segment[1])
    IMUtime = IMUtime[TS:TE]
    iacc = iacc[TS:TE,:]; igyr = igyr[TS:TE,:]
    igyr_det = igyr_det[TS:TE]
    
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
        if Lentries[ii].count(RIMUno):
            edgeang_dwn_gyrR.extend(tmp_edge_dwn)
        else:
            edgeang_dwn_gyrL.extend(tmp_edge_dwn)

# Feedback for the athletes
L_avgedge = np.mean(edgeang_dwn_gyrL)
R_avgedge = np.mean(edgeang_dwn_gyrR)
avgDiff = abs(R_avgedge - L_avgedge)

print('Average Edge Angle: ', round(np.mean(edgeang_dwn_gyrL + edgeang_dwn_gyrR)))
if R_avgedge > L_avgedge:
    print("Skier has ", round(avgDiff), "greater edge angle on right foot")

if L_avgedge > R_avgedge:
    print("Skier has ", round(avgDiff), "greater edge angle on left foot")


