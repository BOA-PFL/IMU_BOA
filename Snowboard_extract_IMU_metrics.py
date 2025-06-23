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
from scipy.integrate import cumulative_trapezoid
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
# import addcopyfighandler
from tkinter import messagebox 


# Obtain IMU signals
fPath = 'Z:\\Testing Segments\\Snow Performance\\2024\\EH_Snowboard_BurtonWrap_Perf_Dec2024\\IMU\\'

Lentries_board = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and (fName.count('04116') or fName.count('03399') or fName.count('04580'))]
Lentries_boot = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('04241')]
        
bindingDat = pd.read_excel('Z:\\Testing Segments\\Snow Performance\\2024\\EH_Snowboard_BurtonWrap_Perf_Dec2024/QualData.xlsx', 'Qual')
bindingDat = bindingDat.iloc[:,:5].dropna()
bindingDat['Subject'] = bindingDat['Subject'].str.replace(' ', '')

# Global variables
# Filtering
gyr_cut = 1

# Debugging variables
debug = 1
save_on = 0

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
    # Set up a 2nd order low pass buttworth filter
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
    max_ang : list (Mx1)
        Maximum toe edge angle (deg)
    min_ang : list (Mx1)
        Maximum heel edge angle (deg)

    """
    
    max_ang = [] # toe edge for board angle 
    min_ang = []
    
    for jj in range(len(turn_idx)-1):
        roll_ang = cumulative_trapezoid(gyr_roll[turn_idx[jj]:turn_idx[jj+1]],t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
        # Remove drift
        slope = roll_ang[-1]/(len(roll_ang)-1)
        drift = (slope*np.ones([len(roll_ang)]))*(np.array(range(len(roll_ang))))
        roll_ang = roll_ang-drift
        max_ang.append(np.max(roll_ang))
        min_ang.append(abs(np.min(roll_ang)))
    
    return(max_ang,min_ang)


badFileList = []

# Variables for first time trial segmentation
st_idx = []
en_idx = []
trial_name = []

# Biomechanical outcomes
boardAng_toe = []
boardAng_heel= []
boot_flex = []

sName = []
cName = []
TrialNo = []
Side = []
TurnTime = []

print('This script assumes that the IMU is placed on the leading/front boot')

for ii in range(6,len(Lentries_board)):
    print(Lentries_board[ii])
    # Extract trial information 
    tmpsName = Lentries_board[ii].split(sep = "-")[1]
    tmpDir = Lentries_board[ii].split(sep = '-')[2] # Regular or Goofy
    tmpConfig = Lentries_board[ii].split(sep = "-")[3]
    tmpTrialNo = Lentries_board[ii].split(sep = "-")[4][0]

    # Import .csv
    Ldf_board = pd.read_csv(fPath + Lentries_board[ii],sep=',', header = 0)
    igyr_board = np.array(Ldf_board.iloc[:,5:8])
    IMUtime_board = np.array(Ldf_board.iloc[:,0])
    
    # Make sure that the boot & board file are for the same trial
    
    Ldf_boot = pd.read_csv(fPath + Lentries_boot[ii], sep = ',', header = 0)
    igyr_boot = np.array(Ldf_boot.iloc[:,5:8])
    IMUtime_boot = np.array(Ldf_boot.iloc[:,0])

    # Convert the time
    IMUtime_board = (IMUtime_board - IMUtime_board[0])*(1e-6)
    IMUtime_boot = (IMUtime_boot - IMUtime_boot[0])*(1e-6)
    
    # Detection signal based on the angular velocity of the snowboard shifting
    # while turning from the heel to toe edges of the snowboard
    igyr_det = filtIMUsig(igyr_board,0.5,IMUtime_board)
    igyr_det = igyr_det[:,1]
   
    #__________________________________________________________________________
    # Trial Segmentation
    if os.path.exists(fPath+Lentries_board[ii]+'TrialSeg.npy'):
        # Load the trial segmentation
        trial_segment = np.load(fPath+Lentries_board[ii]+'TrialSeg.npy', allow_pickle =True)
    else:
        # Segment the trial based on the gyroscope deteciton signal
        trial_segment = delimitTrialIMU(igyr_det)
        # Save the trial segmentation
        np.save(fPath+Lentries_board[ii]+'TrialSeg.npy',trial_segment)
    tmp_st = int(trial_segment[0]); tmp_en = int(trial_segment[1])        
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    IMUtime_board = IMUtime_board[tmp_st:tmp_en]; IMUtime_boot = IMUtime_boot[tmp_st:tmp_en]
    igyr_board = igyr_board[tmp_st:tmp_en,:]; igyr_boot = igyr_boot[tmp_st:tmp_en]
    
    igyr_board = filtIMUsig(igyr_board,gyr_cut,IMUtime_board)
    igyr_boot = filtIMUsig(igyr_boot, gyr_cut, IMUtime_boot)
    
    # IMU alignment: Cross correlate the boot and board signals
    # Note: This assumes that the IMU is placed on the front boot    
    if tmpDir == 'Goofy': # Note: if the IMU is place on the back boot, this will need to be changed to regular
        igyr_boot = -igyr_boot
        
    corr = sig.correlate(igyr_boot[:,2],igyr_board[:,1], mode = 'full')    
        
    lags = sig.correlation_lags(len(igyr_boot[:,2]),len(igyr_board[:,1]),mode='full')
    
    # plt.figure()
    # plt.plot(lags, corr)
    lag = lags[np.argmax(corr)]
    
    if lag > 0:
        igyr_boot = igyr_boot[lag:,:]
        IMUtime_boot = IMUtime_boot[lag:]
        igyr_board = igyr_board[:-lag,:]
        IMUtime_board = IMUtime_board[:-lag]
        
    elif lag < 0:
        igyr_boot = igyr_boot[:lag,:]
        IMUtime_boot = IMUtime_boot[:lag]
        igyr_board = igyr_board[-lag:,:]
        IMUtime_board = IMUtime_board[-lag:]
    
    igyr_det = filtIMUsig(igyr_board,0.5,IMUtime_board)
    igyr_det = igyr_det[:,1]      
    #__________________________________________________________________________
    # Turn detection
    ipeaks,_ = sig.find_peaks(igyr_det, height = 10, distance = 200) # create a function to filter out double peak detections.
    # Filter out the double peak detection
    ii = 0
    ipeaks_up = []
    while ii < len(ipeaks)-1:
        if np.sum(igyr_det[ipeaks[ii]:ipeaks[ii+1]] < -5) > 10:
            # There is a good peak detection
            ipeaks_up.append(ipeaks[ii])
            ii = ii+1
        else:
            # There is not a good detection. Figure out which of the two peaks
            # are higher in magnitude
            if igyr_det[ipeaks[ii]] > igyr_det[ipeaks[ii+1]]:
                ipeaks_up.append(ipeaks[ii])
                ii = ii+2
            else:
                ipeaks_up.append(ipeaks[ii+1])
                ii = ii+2
        
    ipeaks_up = np.array(ipeaks_up)    
    
    # Visualize the peak detection
    answer = True # Defaulting to true: In case "debug" is not used
    if debug == 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(igyr_boot[:,2], label = 'shifted boot gyr')
        plt.plot(igyr_board[:,1], label = 'shifted board gyr')
        plt.legend()
        plt.title('XCorr Check: Shifted Frames:'+str(lag))
        plt.ylabel('Angular Velocity (deg/sec)')
        plt.subplot(1,2,2)
        plt.plot(igyr_det)
        plt.plot(ipeaks_up,igyr_det[ipeaks_up],'bs')
        plt.title('Turn ID Check')
        answer = messagebox.askyesno("Question","Is data clean?") 
        saveFolder = fPath + 'IMU_PeakDetectionPlots'
        
        if answer == True:
            if os.path.exists(saveFolder) == False:
                os.mkdir(saveFolder) 
                
            plt.savefig(saveFolder + '/' + Lentries_board[ii]  +'.png')
         
        plt.close('all')
    
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(Lentries_board[ii])
    
    if answer == True:
        print('Estimating point estimates')
        #______________________________________________________________________
        # Find the edge angles from the gyroscope
        tmp_boardheel, tmp_boardtoe = findEdgeAng_gyr(igyr_board[:,1],IMUtime_board,ipeaks)
        boardAng_toe.extend(tmp_boardtoe)
        boardAng_heel.extend(tmp_boardheel)
        frontAng = bindingDat[bindingDat.Subject == tmpsName].reset_index()['FrontBindingAngle'][0] * np.pi/180 # convert to radians
        
        for jj in range(len(ipeaks)-1):            
            igyr_boardTrim = igyr_board[ipeaks[jj]:ipeaks[jj+1], :]            
            
            # Rotate the Board to the angle of the boot
            # Note: positive rotation for goofy, negative for regular
            if tmpDir == 'Regular':
                rotZ = np.array([[cos(-frontAng),-sin(-frontAng),0], [sin(-frontAng),cos(-frontAng),0], [0,0,1]])
            else:
                rotZ = np.array([[cos(frontAng),-sin(frontAng),0], [sin(frontAng),cos(frontAng),0], [0,0,1]])
        
            igyr_boardRot = (rotZ @ igyr_boardTrim.T).T
                                    
            tmp_boardang=cumulative_trapezoid(igyr_boardRot[:, 1], IMUtime_board[ipeaks[jj]:ipeaks[jj+1]], initial = 0, axis = 0)
            tmp_bootang =cumulative_trapezoid(igyr_boot[ipeaks[jj]:ipeaks[jj+1], 2], IMUtime_boot[ipeaks[jj]:ipeaks[jj+1]], initial = 0, axis = 0)
            
            # Remove drift
            slope_board = tmp_boardang[-1]/(len(tmp_boardang)-1)
            drift_board = (slope_board*np.ones([len(tmp_boardang)]))*(np.array(range(len(tmp_boardang))))
            tmp_boardang = tmp_boardang-drift_board
            
            
            slope_boot = tmp_bootang[-1]/(len(tmp_bootang)-1)
            drift_boot = (slope_boot*np.ones([len(tmp_bootang)]))*(np.array(range(len(tmp_bootang))))
            tmp_bootang = tmp_bootang-drift_boot
            
            tmp_boot_flex = tmp_bootang-tmp_boardang 
            boot_flex.append(abs(np.min(tmp_boot_flex)))
            
            TurnTime.append(IMUtime_board[ipeaks[jj+1]]-IMUtime_board[ipeaks[jj]])
            
            
        # Appending names
        
        sName = sName + [tmpsName]*len(tmp_boardtoe)
        cName = cName + [tmpConfig]*len(tmp_boardtoe)
        TrialNo = TrialNo + [tmpTrialNo]*len(tmp_boardtoe)
        
# Save the outcome metrics
outcomes = pd.DataFrame({'Subject':list(sName),'Config':list(cName),'Order':list(TrialNo), 'TurnTime':list(TurnTime),
                         'BoardAngle_ToeTurns':list(boardAng_toe), 'BoardAngle_HeelTurns':list(boardAng_heel), 
                         'BootFlex':list(boot_flex)})  



outfileName = fPath + '0_IMUOutcomes.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
    
        outcomes.to_csv(outfileName, header=True, index = False)

    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
        
        

