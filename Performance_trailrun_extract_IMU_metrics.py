# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:01:41 2022

@author: Eric.Honert

Notes:
    This code should only be run after extracting metrics from the GPS watch
    
    This code is only good for computing metrics from 1 IMU
    
    Updated footdetection 6/12/24
"""

# Import Libraries
import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2
import scipy
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import addcopyfighandler
from tkinter import messagebox

from IMUFunctions import (delimitTrialIMU, align_fuse_extract_IMU, findRotToLab,
                          filtIMUsig, intp_strides, computeRunSpeedIMU)

# Grab the GPS data for the timing of segments
GPStiming = pd.read_csv('Z:\\Testing Segments\\Outdoor\\TrailRunning\\2024\\EH_Trail_Kailas_Perf_Jun24\\GPS\\CombinedGPS.csv')
# Obtain IMU signals

fPath = 'Z:\\Testing Segments\\Outdoor\\TrailRunning\\2024\\EH_Trail_Kailas_Perf_Jun24\\IMU\\'

save_on = 0
debug = 1

# Right High and Low G accelerometers: note that the gyro is in the low G file
RHentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('04116')] # updated IMU number for infield collections
RLentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('04116')]

# Functions
def estIMU_HS_MS(acc,gyr,t,HS_thresh):
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
    HS_thresh : float/int
        threshold for detecting when heel strikes (foot contacts)

    Returns
    -------
    HS : list
        Heel-strike (foot contact events) indices
    MS : list
        Mid-stance indices

    """
    
    HS_sig = (np.gradient(np.linalg.norm(acc,axis=1),t))**2
    gyr_energy = (np.linalg.norm(gyr,axis=1))**2
    # Create a midstance detection signal
    MS_sig = gyr_energy
    # idx = np.linalg.norm(acc,axis = 1) > 2.5*9.81 # Only want values above 2g as they will be excluded, may need to reduce this threshold
    # MS_sig[idx] = 1e6
    
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = 50 / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    
    # Filter the IMU signals
    acc_filt = np.array([sig.filtfilt(b, a, acc[:,jj]) for jj in range(3)]).T

    window = 200
    jj = 400
    
    HS = []
    
    while jj < len(HS_sig)-1500:
        if HS_sig[jj] > HS_thresh:
            # Find the maximum
            jj = np.argmax(HS_sig[jj:jj+window])+jj         
            HS_idx = np.argmax(acc[jj-window:jj+window,2])+jj-window
            pre_vel = np.trapezoid(acc_filt[HS_idx-150:HS_idx ,2],t[HS_idx-150:HS_idx])
            if pre_vel < -0.5 and acc[HS_idx,2] > 2:
                HS.append(HS_idx)
                jj = jj+500
        jj = jj+1

    # Compute the mid-stance indicies: full "for" loop listed below for debugging
    MS = np.array([(np.argmin(MS_sig[HS[jj]:HS[jj]+int((HS[jj+1]-HS[jj])*0.5)])+HS[jj])  for jj in range(len(HS)-1)]) 
    # MS = []
    # for jj in range(len(HS)-1):
    #     print(jj)
    #     MS.append(np.argmin(MS_sig[HS[jj]+10:HS[jj]+int((HS[jj+1]-HS[jj])*0.2)])+HS[jj]+10)
    
    HS = np.array(HS[:-1])  
        
    
    return [HS,MS]

# Preallocate variables
oSubject = []
oConfig = []
oSesh = []
oLabel = np.array([])
oSide = []
oSpeed = np.array([])

pGyr = []
pJerk = []
rMLacc = []
rIEgyro = []
pIEgyro = []
pAcc = []

badFileList = []

# Filtering frequencies
acc_cut = 50
gyr_cut = 30

# Index through the GPS file as that has all entries possible
for ii in range(0,len(GPStiming)):
    # Find the correct files if there
    GPSstr = GPStiming.Subject[ii] + '-' + GPStiming.Config[ii]  + '-' + str(GPStiming.Sesh[ii])
    
    # Check to make sure there is an IMU trial for for the selected GPS trial 
    GoodTrial = 0; Rtrial = []
    for jj, entry in enumerate(RLentries):
        if GPSstr in entry:
            GoodTrial = 1
            Rtrial = jj
       
    # Conditional statement
    if GoodTrial == 1:
        print(RLentries[Rtrial])
        # Load the trials here
        RLdf = pd.read_csv(fPath + RLentries[Rtrial],sep=',', header = 0)
        RHdf = pd.read_csv(fPath + RHentries[Rtrial],sep=',', header = 0)
        
        # Align & fuse the high and low-g accelerometer signals. Extract time
        # and gyro as well.
        [Rtime,Racc,Rgyr] = align_fuse_extract_IMU(RLdf,RHdf)
        #______________________________________________________________________
        # Trial Segmentation
        if os.path.exists(fPath+RLentries[ii]+'TrialSeg.npy'):
            # Load the trial segmentation
            trial_segment = np.load(fPath+RLentries[ii]+'TrialSeg.npy', allow_pickle =True)
        else:
            # Segment the trial based on the gyroscope deteciton signal
            trial_segment = delimitTrialIMU(Racc[:,2])
            # Save the trial segmentation
            np.save(fPath+RLentries[ii]+'TrialSeg.npy',trial_segment)
        #______________________________________________________________________
        # Use only the data from the pre-selected region
        TS = int(trial_segment[0]); TE = int(trial_segment[1])
        Rtime = Rtime[TS:TE]
        Racc = Racc[TS:TE,:]; Rgyr = Rgyr[TS:TE,:]
        #______________________________________________________________________        
        # Convert the time
        Rtime = (Rtime - Rtime[0])*(1e-6)
        
        # Identify foot contact & midstance
        [RHS,RMS] = estIMU_HS_MS(Racc,Rgyr,Rtime,5e7)
        # Generally, the first 3 detected HS are from hops (manually checked as well)
        R_start = Rtime[RHS[2]]
        RHS = RHS[3:]; RHS_t = Rtime[RHS]
        RMS = RMS[3:]
        
        # Find good strides
        RGS = np.where((np.diff(RHS) > 0.5)*(np.diff(RHS_t) < 1.5))[0]
        answer = True # Defaulting to true: In case "debug" is not used
        if debug == 1:
            # Examine foot contact detections:
            plt.plot(Rtime,Racc[:,2])
            plt.plot(RHS_t,Racc[RHS,2],'ro')
            plt.plot(RHS_t[RGS],Racc[RHS[RGS],2],'ko')
            plt.ylabel('Vertical Acceleration [m/s^2]')
            plt.xlabel('Time [sec]')
            # Look at acceleration and gyro for each different section
            GS_up = []; GS_top = []; GS_dwn = []
            for jj in RGS:
                if RHS_t[jj] < float(GPStiming.EndS1[ii]+R_start):
                    GS_up.append(jj)
                elif RHS_t[jj] > float(GPStiming.StartS2[ii]+R_start) and RHS_t[jj] < float(GPStiming.EndS2[ii]+R_start):
                    GS_top.append(jj)
                elif RHS_t[jj] > float(GPStiming.StartS3[ii]+R_start):
                    GS_dwn.append(jj)
            
            plt.figure(101)
            plt.subplot(2,3,1)
            plt.plot(intp_strides(Racc[:,2],RHS, GS_up))
            plt.ylabel('Vertical Acceleration [m/s^2]')
            plt.title('Uphill')
            
            plt.subplot(2,3,2)
            plt.plot(intp_strides(Racc[:,2],RHS, GS_top))
            plt.title('Top')
            
            plt.subplot(2,3,3)
            plt.plot(intp_strides(Racc[:,2],RHS, GS_dwn))
            plt.title('Downhill')
            
            plt.subplot(2,3,4)
            plt.plot(intp_strides(Rgyr[:,1],RHS, GS_up))
            plt.ylabel('In/Ev Gyro [deg/s]')
            plt.xlabel('% Stride')
            
            plt.subplot(2,3,5)
            plt.plot(intp_strides(Rgyr[:,1],RHS, GS_top))
            plt.xlabel('% Stride')
            
            plt.subplot(2,3,6)
            plt.plot(intp_strides(Rgyr[:,1],RHS, GS_dwn))
            plt.xlabel('% Stride')
            plt.tight_layout()
            
            saveFolder = fPath + 'IMU_Plots'
            
            if answer == True:
                if os.path.exists(saveFolder) == False:
                    os.mkdir(saveFolder)  
                plt.savefig(saveFolder + '/' + RLentries[ii].split(sep="_")[0]  +'.png')
            
            
            answer = messagebox.askyesno("Question","Is data clean?")
            plt.close('all')
        
            if answer == False:
                print('Adding file to bad file list')
                badFileList.append(RLentries[ii])
            
        if answer == True:
            print('Computing point estimates')
            # Compute the step-by-step running speed
            Rspeed = computeRunSpeedIMU(Racc,Rgyr,RHS,RMS,RGS,Rtime)
            RGS = RGS[0:-2]

            # Filter the IMU signals
            Racc = filtIMUsig(Racc,acc_cut,Rtime)
            Rgyr = filtIMUsig(Rgyr,gyr_cut,Rtime)
            # Compute stride metrics here
            Rjerk = np.linalg.norm(np.array([np.gradient(Racc[:,jj],Rtime) for jj in range(3)]),axis=0)
            Racc_mag = np.linalg.norm(Racc,axis=1)
            for jj in RGS:
                pJerk.append(np.max(Rjerk[RHS[jj]:RHS[jj+1]]))
                pAcc.append(np.max(Racc_mag[RHS[jj]:RHS[jj+1]]))
                pGyr.append(np.abs(np.min(Rgyr[RHS[jj]:RHS[jj+1],1])))
                rMLacc.append(np.max(Racc[RHS[jj]:RHS[jj+1],1])-np.min(Racc[RHS[jj]:RHS[jj+1],1]))
                appTO = round(0.2*(RHS[jj+1]-RHS[jj])+RHS[jj])
                rIEgyro.append(np.max(Rgyr[RHS[jj]:appTO,2])-np.min(Rgyr[RHS[jj]:appTO,2]))
                pIEgyro.append(np.abs(np.min(Rgyr[RHS[jj]:appTO,2])))
            
            # Create labels for saving data
            Rlabel = np.array([0]*len(RGS))
            # Uphill label
            idx = RHS_t[RGS] < float(GPStiming.EndS1[ii]+R_start)
            Rlabel[idx] = 1
            # Top label
            idx = (RHS_t[RGS] > float(GPStiming.StartS2[ii]+R_start))*(RHS_t[RGS] < float(GPStiming.EndS2[ii]+R_start))
            Rlabel[idx] = 2
            # Bottom label
            idx = RHS_t[RGS] > float(GPStiming.StartS3[ii]+R_start)
            Rlabel[idx] = 3
            
            # Appending
            oSubject = oSubject + [GPStiming.Subject[ii]]*len(RGS)
            oConfig = oConfig + [GPStiming.Config[ii]]*len(RGS)
            oSesh = oSesh + [GPStiming.Sesh[ii]]*len(RGS)
            oLabel = np.concatenate((oLabel,Rlabel),axis = None)
            oSide = oSide + ['R']*len(RGS)
            oSpeed = np.concatenate((oSpeed,Rspeed),axis = None)         
        else:
            print('No IMU Trial found')
    
    
    # Clear variables
    RHS = []; RGS = []
    
        
outcomes = pd.DataFrame({'Subject':list(oSubject), 'Side':list(oSide), 'Config': list(oConfig),'Sesh': list(oSesh),
                          'Label':list(oLabel), 'pJerk':list(pJerk), 'pAcc':list(pAcc), 'pGyr':list(pGyr),'rMLacc':list(rMLacc),'rIEgyro':list(rIEgyro),'pIEgyro':list(pIEgyro),'imuSpeed':list(oSpeed)})

if save_on == 1:
    outcomes.to_csv(fPath+'IMUmetrics.csv',header=True)
elif save_on == 2:
    outcomes.to_csv(fPath+'IMUmetrics.csv',mode = 'a',header=False)


