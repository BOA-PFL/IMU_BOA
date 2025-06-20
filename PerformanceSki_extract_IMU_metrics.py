# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:04:52 2022

Code to examine angles provided by vicon IMUs

@author: Eric.Honert
"""
# Import Libraries
import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import addcopyfighandler
from tkinter import messagebox

# Obtain IMU signals
fPath = 'Z:/Testing Segments/Snow Performance/2025_Mech_AlpineRace/IMU/'

# Global variables
# Filtering
gyr_cut = 6

# Debugging variables
debug = 0
save_on = 0

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
    Using the gyroscope, compute the maximum edge angles by integrating the
    roll angular velocity and use a linear correction to account for
    integration drift

    Parameters
    ----------
    gyr_roll : numpy array (Nx1)
        Roll angular velocity (deg/sec)
    t : numpy array (Nx1)
        time (sec)
    turn_idx : numpy array (Mx1)
        turn indicies

    Returns
    -------
    edgeang_dwn : list (Mx1)
        Maximum downhill ski edge angle (deg)
    edgeang_up : list (Mx1)
        Maximum uphill ski edge angle (deg)
    rate_edgeang_dwn : list (Mx1)
        Average anuglar velocity while attaining to maximum downhill edge angle (deg/sec)        
    timePeak : list (Mx1)
        Time to maximum downhill edge angle (sec)

    """
    
    edgeang_dwn = []
    rate_edgeang_dwn = []
    timePeak = []
    edgeang_up = []
    
    # Provide interpolation variable for debugging purposes
    intp_var = np.zeros((101,len(turn_idx)-1))
    
    # Loop through all of the turns and integrate the roll angular velocity
    for jj in range(len(turn_idx)-1):
        roll_angvel = gyr_roll[turn_idx[jj]:turn_idx[jj+1]]
        t_turn = t[turn_idx[jj]:turn_idx[jj+1]]
        roll_ang = cumtrapz(roll_angvel,t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
        # Remove integration drift
        slope = roll_ang[-1]/(len(roll_ang)-1)
        drift = (slope*np.ones([len(roll_ang)]))*(np.array(range(len(roll_ang))))
        roll_ang = roll_ang-drift
        # Store the max edge angles
        edgeang_dwn.append(np.max(roll_ang))
        edgeang_up.append(abs(np.min(roll_ang)))
        # Time to peak edge angle
        idx = np.argmax(roll_ang)
        timePeak.append(t_turn[idx]-t_turn[0])
        # Average rate to get to peak edge angle
        rate_edgeang_dwn.append(np.mean(roll_angvel[0:idx]))
        
        # Store the interpolated time-continous for debugging purposes
        f = scipy.interpolate.interp1d(np.arange(0,len(roll_ang)),roll_ang)
        intp_var[:,jj] = f(np.linspace(0,len(roll_ang)-1,101))
        
    
    return(edgeang_dwn,edgeang_up,rate_edgeang_dwn,timePeak)


badFileList = []

# Variables for first time trial segmentation
st_idx = []
en_idx = []
trial_name = []

# Biomechanical outcomes
edgeang_dwn_gyr = []
RAD_dwn = [] # Rate of angle development
edgeangt_dwn_gyr = []
edgeang_up_gyr = []

sName = []
cName = []
TrialNo = []
Side = []
TurnTime = []
   
# High and Low G accelerometers: note that the gyro is in the low G file
Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv')]
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv')]


for ii in range(len(Lentries)):
    # Grab the .csv files
    print(Lentries[ii])
    # Extract trial information
    tmpsName = Lentries[ii].split(sep = "-")[1]
    tmpConfig = Lentries[ii].split(sep = "-")[2]
    tmpTrialNo = Lentries[ii].split(sep = "-")[3][0]
    
    Ldf = pd.read_csv(fPath + Lentries[ii],sep=',', header = 0)
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
    if os.path.exists(fPath+Lentries[ii]+'TrialSeg.npy'):
        # Load the trial segmentation
        trial_segment = np.load(fPath+Lentries[ii]+'TrialSeg.npy', allow_pickle =True)
    else:
        # Segment the trial based on the gyroscope deteciton signal
        trial_segment = delimitTrialIMU(igyr_det)
        # Save the trial segmentation
        np.save(fPath+Lentries[ii]+'TrialSeg.npy',trial_segment)
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    TS = int(trial_segment[0]); TE = int(trial_segment[1])
    IMUtime = IMUtime[TS:TE]
    igyr = igyr[TS:TE,:]
    igyr_det = igyr_det[TS:TE]
    
    #__________________________________________________________________________
    # Turn segmentation: Find when the turn is happening
    ipeaks,_ = sig.find_peaks(igyr_det, height = 15, distance = 300)
    # Visualize the peak detection
    answer = True # Defaulting to true: In case "debug" is not used
    if debug == 1:
        plt.figure()
        plt.plot(igyr_det)
        plt.plot(ipeaks,igyr_det[ipeaks],'bs')
        answer = messagebox.askyesno("Question","Is data clean?")
        plt.close('all')
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(Lentries[ii])
    
    if answer == True:
        print('Estimating point estimates')
        #______________________________________________________________________
        # Find the edge angles from the gyroscope
        igyr = filtIMUsig(igyr,gyr_cut,IMUtime)
        tmp_edge_dwn,tmp_edge_up,tmp_RAD_dwn,tmp_edgeangt_dwn_gyr = findEdgeAng_gyr(igyr[:,2],IMUtime,ipeaks)
        edgeang_dwn_gyr.extend(tmp_edge_dwn)
        edgeang_up_gyr.extend(tmp_edge_up)
        RAD_dwn.extend(tmp_RAD_dwn)
        edgeangt_dwn_gyr.extend(tmp_edgeangt_dwn_gyr)
        
        # Appending names
        if Lentries[ii].count(RIMUno):
            Side = Side + ['R']*len(tmp_edge_dwn)
        else:
            Side = Side + ['L']*len(tmp_edge_dwn)
        sName = sName + [tmpsName]*len(tmp_edge_dwn)
        cName = cName + [tmpConfig]*len(tmp_edge_dwn)
        TrialNo = TrialNo + [tmpTrialNo]*len(tmp_edge_dwn)
        

# Save the outcome metrics
outcomes = pd.DataFrame({'Subject':list(sName),'Config':list(cName),'Order':list(TrialNo),'Side': list(Side),
                                 'edgeang_dwn_gyr':list(edgeang_dwn_gyr), 'edgeang_up_gyr':list(edgeang_up_gyr),
                                 'RAD_dwn':list(RAD_dwn), 'edgeangt_dwn_gyr':list(edgeangt_dwn_gyr)
                                 })  

outfileName = fPath+'0_IMUOutcomes.csv'
if save_on == 1:
    outcomes.to_csv(outfileName, header=True, index = False)

