# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:01:41 2022

@author: Eric.Honert

Code created to compute walking/running speed and extract metrics from IMUs

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

from IMUFunctions import (align_fuse_extract_IMU, findRotToLab, filtIMUsig,
                          intp_strides, computeRunSpeedIMU)


# Obtain IMU signals
fPath = 'C:\\Users\\max.ferguson\\OneDrive - BOA Technology Inc\\PFL Team - General\\Testing Segments\\Outdoor\\TrailRunning\\2026_Performance_Kailas\\IMU\\'

save_on = 1
debug = 1

# High and Low G accelerometers: note that the gyro is in the low G file
# Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') ] 
# Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv')] 

Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and ('04241') in fName] 
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and ('04241') in fName] 

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
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = 50 / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    
    # Filter the IMU signals
    acc_filt = np.array([sig.filtfilt(b, a, acc[:,jj]) for jj in range(3)]).T
    
    HS_sig = (np.gradient(np.linalg.norm(acc_filt,axis=1),t))**2
    # HS_sig = (np.gradient(acc_filt[:,2],t))**2
    gyr_energy = (np.linalg.norm(gyr,axis=1))**2
    # Create a midstance detection signal
    idx = np.linalg.norm(acc,axis = 1) > 8*9.81 # Only want values above 2g as they will be excluded, may need to reduce this threshold
    MS_sig = gyr_energy
    MS_sig[idx] = 1e6
    window = 100
    jj = 400
    
    HS = []
    
    while jj < len(HS_sig)-1500:
        if HS_sig[jj] > HS_thresh:
            # Find the maximum
            jj = np.argmax(HS_sig[jj:jj+window])+jj         
            HS_idx = np.argmax(acc[jj-window:jj+window,2])+jj-window
            pre_vel = np.trapezoid(acc_filt[HS_idx-150:HS_idx ,2],t[HS_idx-150:HS_idx])
            if pre_vel < 1 and acc[HS_idx,2] > 5:
                HS.append(HS_idx)
                jj = jj+300
        jj = jj+1
              
    # Compute the mid-stance indicies: full "for" loop listed below for debugging
    MS = np.array([(np.argmin(MS_sig[HS[jj]+10:HS[jj]+int((HS[jj+1]-HS[jj])*0.2)])+HS[jj]+10)  for jj in range(len(HS)-1)]) 
    # MS = []
    # for jj in range(len(HS)-1):
    #     print(jj)
    #     MS.append(np.argmin(MS_sig[HS[jj]+10:HS[jj]+int((HS[jj+1]-HS[jj])*0.2)])+HS[jj]+10)
    HS = np.array(HS[:-1])
        
    
    return [HS,MS]

def plotStrides(inputAcc, inputGy, inputLandings, goodLandings):
    """
    Function to plot interpolated strides for vertical acceleration and
    inversion/eversion gyroscope
    Function dependances: use intp_strides to generate interpolated strides

    Parameters
    ----------
    inputAcc : numpy array
        Input acceleration. Ex: Vertical acceleration (Nx1)
    inputGy : numpy array
        Input gyroscope signal. Ex: inversion/eversion velocity (Nx1) 
    inputLandings : numpy array
        Detected landings. From IMU: use estIMU_HS_MS (Mx1)
    goodLandings : numpy array
        Filtered landings usually based on stride time (Qx1)

    Returns
    -------
    None.

    """
    # plots interpolated stride, acc & gyro
    plt.figure(101)
    plt.subplot(1,2,1) # first plot
    plt.plot(intp_strides(inputAcc,inputLandings, goodLandings), 'k')
    plt.ylabel('Vertical Acceleration [m/s^2]')
    plt.subplot(1,2,2) # second plot
    plt.plot(intp_strides(inputGy,inputLandings, goodLandings), 'k')
    plt.ylabel('In/Ev Angular Velocity [deg/s]')
    plt.tight_layout()
    plt.show()

# Storing Variables
oSubject = []
oConfig = []
oSesh = []
oMovement = []

pGyr = []
pAcc = []
pJerk = []
rMLacc = []
rIEgyro = []
pIEgyro = []
pEgyro = []
pIgyro = []
imuSpeed = []

badFileList = []

# Filtering frequencies
acc_cut = 50
gyr_cut = 30

# Index through the low-g files
for ii in range(len(Lentries)):
    print(Lentries[ii])
    # Load the trials here
    Ldf = pd.read_csv(fPath + Lentries[ii],sep=',', header = 0)
    Hdf = pd.read_csv(fPath + Hentries[ii],sep=',', header = 0)
    # Save trial information
    Subject = Lentries[ii].split(sep = "-")[0]
    Config = Lentries[ii].split(sep="-")[1] 
    Movement = Lentries[ii].split(sep="-")[2]
    Sesh = Lentries[ii].split(sep="-")[3][0]
    
    # Fuse the low-g & high-g accelerometers
    [IMUtime,iacc,igyr] = align_fuse_extract_IMU(Ldf,Hdf)
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)        
    # Identify foot contact & midstance
    [iHS,iMS] = estIMU_HS_MS(iacc,igyr,IMUtime,1e4)
    
    # Examine where the start of the trial is by the 3 jumps
    # There should seem to be 2 "short" strides followed by a pause
    approx_CT = np.diff(iHS)
    iHS_t = IMUtime[iHS]
    
    # Counters
    jc = 0  # jump counter
    stc = 0 # start trial counter
    jj = 0
    up_thresh = 3e6 # This threshold should be modulated based on running or walking
    
    
    # Algorithm to detect 3 hops - may need to be updated
    # while stc == 0:
    #     if approx_CT[jj] < 1500:
    #         jc = jc+1
    #     if jc >= 2 and approx_CT[jj] > 2000:
    #         idx = (iHS_t > (iHS_t[jj] + 5))
    #         iHS = iHS[idx]
    #         iHS_t = iHS_t[idx]
    #         iMS = iMS[idx]
    #         stc = 1
        
    #     jj = jj+1
        
    #     if jj > 10:
    #         up_thresh = up_thresh - 2e6
    #         [iHS,iMS] = estIMU_HS_MS(iacc,igyr,IMUtime,up_thresh)
    #         approx_CT = np.diff(iHS)
    #         iHS_t = IMUtime[iHS]
    #         jj = 0
        
    
    # Exclusion criteria for good strides: Based on step length: for bad detections
    iGS = np.where((np.diff(iHS_t) > .75)*(np.diff(iHS_t) < 2))[0]
    iHS = iHS[iGS]; iMS = iMS[iGS]
    iHS_t = IMUtime[iHS]
    
    # Further exclude strides to ensure that only consistent strides are used
    iGS = np.where((np.diff(iHS_t) > .75)*(np.diff(iHS_t) < 2))[0]
    
    # Debugging: Creation of dialog box for looking where foot contact are accurate
    answer = True # Defaulting to true: In case "debug" is not used    
    if debug == 1:
        plotStrides(iacc[:,2], igyr[:,1], iHS, iGS)
        answer = messagebox.askyesno("Question","Is data clean?")
        saveFolder = fPath + 'IMU_Plots'
        
        if answer == True:
            if os.path.exists(saveFolder) == False:
                os.mkdir(saveFolder)  
            plt.savefig(saveFolder + '/' + Lentries[ii].split(sep="_")[0]  +'.png')
        
        plt.figure()
        plt.plot(IMUtime,iacc[:,2])
        plt.plot(iHS_t,iacc[iHS,2],'ro')
        plt.plot(iHS_t[iGS],iacc[iHS[iGS],2],'ko')
        plt.ylabel('Vertical Acceleration [m/s^2]')
        plt.xlabel('Time [sec]')
        
        answer = messagebox.askyesno("Question","Is data clean?")
        
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(Lentries[ii])
        
    # plt.close('all')
          
    if answer == True:
        print('Estimating point estimates')
        # Compute IMU running speed
        imuSpeed = np.concatenate((imuSpeed,computeRunSpeedIMU(iacc,igyr,iHS,iMS,iGS,IMUtime)),axis = None)
        # Filter the IMU signals
        iacc = filtIMUsig(iacc,acc_cut,IMUtime)
        igyr = filtIMUsig(igyr,gyr_cut,IMUtime)
        # Compute stride metrics here
        jerk = np.linalg.norm(np.array([np.gradient(iacc[:,jj],IMUtime) for jj in range(3)]),axis=0)
        AccMag = np.linalg.norm(iacc,axis=1)
        for jj in iGS:
            pJerk.append(np.max(jerk[iHS[jj]:iHS[jj+1]]))
            pAcc.append(np.max(AccMag[iHS[jj]:iHS[jj+1]]))
            pGyr.append(np.abs(np.min(igyr[iHS[jj]:iHS[jj+1],1])))
            rMLacc.append(np.max(iacc[iHS[jj]:iHS[jj+1],1])-np.min(iacc[iHS[jj]:iHS[jj+1],1]))
            appTO = round(0.2*(iHS[jj+1]-iHS[jj])+iHS[jj])
            rIEgyro.append(np.max(igyr[iHS[jj]:appTO,2])-np.min(igyr[iHS[jj]:appTO,2]))
            # Assuming this is the left foot
            pIEgyro.append(np.max(igyr[iHS[jj]:appTO,2]))
            pEgyro.append(np.abs(np.min(igyr[iHS[jj]:appTO,2])))
            pIgyro.append(-np.max(igyr[iHS[jj]:appTO,2]))               # Check directionality - use commented version if they are negative versions of expected values
            #pEgyro.append(np.max(igyr[iHS[jj]:appTO,2]))
            #pIgyro.append(np.min(igyr[iHS[jj]:appTO,2]))     # already negative
            
        # Appending
        oSubject = oSubject + [Subject]*len(iGS)
        oConfig = oConfig + [Config]*len(iGS) 
        oMovement = oMovement + [Movement]*len(iGS)
        oSesh = oSesh + [Sesh]*len(iGS)
    
    # Clear variables
    iHS = []; iGS = []
    
outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig), 'Movement':list(oMovement),
                         'Order': list(oSesh), 'pJerk':list(pJerk),'pAcc':list(pAcc), 'pGyr':list(pGyr),
                           'rMLacc':list(rMLacc),'rIEgyro':list(rIEgyro),'pIEgyro':list(pIEgyro), 'pIgyro':list(pIgyro),'pEgyro':list(pEgyro),'imuSpeed':list(imuSpeed)})




if save_on == 1:
    outcomes.to_csv(fPath+'0_Trail_CompIMUmetrics.csv',header=True,index=False, mode = 'a')


