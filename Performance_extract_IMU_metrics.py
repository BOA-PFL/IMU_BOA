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

from IMUFunctions import (estIMU_FC_MS_walking, align_fuse_extract_IMU, findRotToLab, filtIMUsig,
                          intp_strides, computeRunSpeedIMU)


# Obtain IMU signals
fPath = 'C:\\Users\\max.ferguson\\OneDrive - BOA Technology Inc\\PFL Team - General\\Testing Segments\\Outdoor\\TrailRunning\\2026_Performance_Kailas\\IMU\\'

save_on = 1

# Foot contact detection (walking). These were the defaults of the local
# detector this script used to carry; they are named here so a change is visible
# in the script rather than buried in IMUFunctions.
fc_jerk_thresh = 1e5    # squared jerk of ||acc|| needed to seed a candidate
fc_min_gap = 0.90       # s, shortest time between successive foot contacts
debug = 1

# High and Low G accelerometers: note that the gyro is in the low G file
# Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') ] 
# Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv')] 

Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and ('04241') in fName] 
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and ('04241') in fName] 

# Functions
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
        Detected landings. From IMU: use estIMU_FC_MS_walking (Mx1)
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
    [iHS,iMS] = estIMU_FC_MS_walking(iacc,igyr,IMUtime,jerk_thresh=fc_jerk_thresh,
                                     min_gap=fc_min_gap)
    
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
    #         [iHS,iMS] = estIMU_FC_MS_walking(iacc,igyr,IMUtime,jerk_thresh=up_thresh)
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


