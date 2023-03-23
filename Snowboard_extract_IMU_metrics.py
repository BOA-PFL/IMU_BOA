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
from tkinter import messagebox

# Obtain IMU signals
fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/Snow Performance/PFLMechanistic_StepOn_March2023/IMUData/'

# Global variables
# Filtering
acc_cut = 1
gyr_cut = 1

# Debugging variables
debug = 1
save_on = 0

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

def findEdgeAng_gyr(gyr_roll,t,turn_idx):
    
    max_ang = [] # toe edge for board angle 
    min_ang = []
    
    for jj in range(len(turn_idx)-1):
        roll_ang = cumtrapz(gyr_roll[turn_idx[jj]:turn_idx[jj+1]],t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
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

# Load in the trial segmentation variable if it is in the directory
if os.path.exists(fPath+'TrialSeg.npy') == True:
    trial_segment_old = np.load(fPath+'TrialSeg.npy')
    trial_name = np.ndarray.tolist(trial_segment_old[0,:])
    st_idx = np.ndarray.tolist(trial_segment_old[1,:])
    en_idx = np.ndarray.tolist(trial_segment_old[2,:])
    
# High and Low G accelerometers: note that the gyro is in the low G file
Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv')]
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv')]

Hentries_board = []
Hentries_boot = []
for h in Hentries: 
    if '03399' in h:
        Hentries_board.append(h)
    elif '03391' in h:
        Hentries_boot.append(h)
        
Lentries_board = []
Lentries_boot = []
for l in Lentries: 
    if '03399' in l:
        Lentries_board.append(l)
    elif '03391' in l:
        Lentries_boot.append(l)
        



for ii in range(0,len(Lentries_board)):
    # Grab the .csv files
    
    ii = 0
    print(Lentries_board[ii])
    # Extract trial information
    tmpsName = Lentries_board[ii].split(sep = "-")[0]
    tmpDir = Lentries_board[ii].split(sep = '-')[1]
    tmpConfig = Lentries_board[ii].split(sep = "-")[2]
    tmpTrialNo = Lentries_board[ii].split(sep = "-")[3][0]
    
    Ldf_board = pd.read_csv(fPath + Lentries_board[ii],sep=',', header = 0)
    Ldf_boot = pd.read_csv(fPath + Lentries_boot[ii], sep = ',', header = 0)
    Hdf_board = pd.read_csv(fPath + Hentries_board[ii],sep=',', header = 0)
    Hdf_boot = pd.read_csv(fPath + Hentries_boot[ii], sep = ',', header = 0 )

    # Combine the IMU data
    [IMUtime_board,iacc_board,igyr_board,iang_board] = align_fuse_extract_IMU_angles(Ldf_board,Hdf_board)
    [IMUtime_boot, iacc_boot, igyr_boot, iang_boot] = align_fuse_extract_IMU_angles(Ldf_boot, Hdf_boot)
    # Convert the time
    IMUtime_board = (IMUtime_board - IMUtime_board[0])*(1e-6)
    IMUtime_boot = (IMUtime_boot - IMUtime_boot[0])*(1e-6)
       
          
        
    
    igyr_det = filtIMUsig(igyr_board,0.5,IMUtime_board)
    igyr_det = igyr_det[:,1]
    
    #__________________________________________________________________________
    # Trial Segmentation
    if Lentries_board[ii] in trial_name:
        # If the trial was previously segmented, grab the appropriate start/end points
        idx = trial_name.index(Lentries_board[ii])
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
        trial_name.append(Lentries_board[ii])
    #__________________________________________________________________________
    # Use only the data from the pre-selected region
    IMUtime_board = IMUtime_board[tmp_st:tmp_en]; IMUtime_boot = IMUtime_boot[tmp_st:tmp_en]
    iang_board = iang_board[tmp_st:tmp_en,:]; iang_boot = iang_boot[tmp_st:tmp_en]
    iacc_board = iacc_board[tmp_st:tmp_en,:]; iacc_boot = iacc_boot[tmp_st:tmp_en]
    igyr_board = igyr_board[tmp_st:tmp_en,:]; igyr_boot = igyr_boot[tmp_st:tmp_en]
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
        igyr_board = filtIMUsig(igyr_board,gyr_cut,IMUtime_board)
        igyr_boot = filtIMUsig(igyr_boot, gyr_cut, IMUtime_boot)
        tmp_boardtoe, tmp_boardheel = findEdgeAng_gyr(igyr_board[:,1],IMUtime_board,ipeaks)
        boardAng_toe.extend(tmp_boardtoe)
        boardAng_heel.extend(tmp_boardheel)
        
        iacc_board = filtIMUsig(iacc_board, acc_cut, IMUtime_board)
        
        # board_angle = []
        # boot_angle = []
        
        #roll_ang = cumtrapz(gyr_roll[turn_idx[jj]:turn_idx[jj+1]],t[turn_idx[jj]:turn_idx[jj+1]],initial=0,axis=0)
        for jj in range(len(ipeaks)-1):
            tmp_boardang=cumtrapz(igyr_board[ipeaks[jj]:ipeaks[jj+1], 1], IMUtime_board[ipeaks[jj]:ipeaks[jj+1]], initial = 0, axis = 0)
            tmp_bootang =cumtrapz(igyr_boot[ipeaks[jj]:ipeaks[jj+1], 2], IMUtime_boot[ipeaks[jj]:ipeaks[jj+1]], initial = 0, axis = 0)
            
            # Remove drift
            slope_board = tmp_boardang[-1]/(len(tmp_boardang)-1)
            drift_board = (slope_board*np.ones([len(tmp_boardang)]))*(np.array(range(len(tmp_boardang))))
            tmp_boardang = tmp_boardang-drift_board
            # board_angle.extend(tmp_boardang)
            
            slope_boot = tmp_bootang[-1]/(len(tmp_bootang)-1)
            drift_boot = (slope_boot*np.ones([len(tmp_bootang)]))*(np.array(range(len(tmp_bootang))))
            tmp_bootang = tmp_bootang-drift_boot
            # boot_angle.extend(tmp_bootang)
            
            tmp_boot_flex = tmp_bootang-tmp_boardang 
            boot_flex.append(np.max(tmp_boot_flex))
            
        # Appending names
        
        sName = sName + [tmpsName]*len(tmp_boardtoe)
        cName = cName + [tmpConfig]*len(tmp_boardtoe)
        TrialNo = TrialNo + [tmpTrialNo]*len(tmp_boardtoe)
        

# Save the trial segmentation
trial_segment = np.array([trial_name,st_idx,en_idx])
np.save(fPath+'TrialSeg.npy',trial_segment)

# Save the outcome metrics
outcomes = pd.DataFrame({'Subject':list(sName),'Config':list(cName),'TrialNo':list(TrialNo),'Side': list(Side),
                                 'BoardAngle_ToeTurns':list(boardAng_toe), 'BoardAngle_HeelTurns':list(boardAng_heel), 'BootFlex':list(boot_flex)
                                 })  

if save_on == 1:
    outcomes.to_csv(fPath+'IMUOutcomes_FBS.csv', header=True, index = False)



