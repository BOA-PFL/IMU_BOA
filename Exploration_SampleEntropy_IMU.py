# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:40:58 2023

@author: Kate.Harrison
"""

### Import packages

import os
import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2
from matplotlib import pyplot as plt
import nolds
import statsmodels.api as sm
from scipy.integrate import cumtrapz
import math
import scipy
import scipy.signal as sig
from scipy.signal import argrelextrema





save_on = 1

### Functions


def align_fuse_extract_IMU(LGdat,HGdat):
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

    """
    # LGdat: low-g dataframe (contains accelerometer and gyroscope)
    # HGdat: high-g data frame (contains only accelerometer)
    
    # Need to align the data
    # 1: get the data collection frequency from the low-g acclerometer
    
   
    acc_lg = np.array(LGdat.iloc[:,2:5])
    gyr = np.array(LGdat.iloc[:,5:])
    
    dum = np.array(HGdat.iloc[:,2:])
    
    HGtime = np.array(HGdat.iloc[:,0])
    LGtime = np.array(LGdat.iloc[:,0])
    
    idx = (LGtime < np.max(HGtime))*(LGtime > np.min(HGtime))
    LGtime = LGtime[idx]
    acc_lg = acc_lg[idx,:]
    gyr = gyr[idx,:]
    
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
        resamp_HG = resamp_HG[:-lag,:]
        
    elif lag < 0:
        LGtime = LGtime[:lag]
        gyr = gyr[:lag,:]
        acc_lg = acc_lg[:lag,:]
        resamp_HG = resamp_HG[-lag:,:]
    
    acc = acc_lg
    
    # Find when the data is above/below 16G and replace with high-g accelerometer
    for jj in range(3):
        idx = np.abs(acc[:,jj]) > (9.81*16-0.1)
        acc[idx,jj] = resamp_HG[idx,jj]
    
    return [LGtime,acc,gyr]

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
    gyr_energy = (np.linalg.norm(gyr,axis=1))**2
    # Create a midstance detection signal
    idx = np.linalg.norm(acc,axis = 1) > 2.5*9.81 # Only want values above 2g as they will be excluded, may need to reduce this threshold
    MS_sig = gyr_energy
    MS_sig[idx] = 1e6
    
    
    window = 200
    jj = 200
    
    HS = []
    # MS = []
    
    while jj < len(HS_sig)-1000:
        if HS_sig[jj] > HS_thresh:
            # Find the maximum
            HS_idx = np.argmin(acc[jj-window:jj+window,0])
            HS.append(jj-window+HS_idx)
            jj = jj+500
        jj = jj+1
            
    # Compute the mid-stance indicies
    MS = np.array([(np.argmin(MS_sig[HS[jj]+10:HS[jj]+int((HS[jj+1]-HS[jj])*0.2)])+HS[jj]+10)  for jj in range(len(HS)-1)]) 
    # MS = []
    # for jj in range(len(HS)-1):
    #     print(jj)
    #     MS.append(np.argmin(MS_sig[HS[jj]+10:HS[jj]+int((HS[jj+1]-HS[jj])*0.2)])+HS[jj]+10)
    HS = np.array(HS[:-1])
        
    
    return [HS,MS]

def findRotToLab(accel_vec):
    """
    Function to find the rotation of the foot flat acceleration vector to the 
    defined lab gravity coordinate system.

    Parameters
    ----------
    accel_vec : numpy array
        3x1 vector of the x,y,z acceleration at foot flat

    Returns
    -------
    theta : numpy array
        3x1 vector for the rotation from the foot flat acceleration to the lab
        coordinate system

    """
    iGvec = accel_vec
    iGvec = iGvec/np.linalg.norm(iGvec)
    # Define the lab coordinate system gravity vector
    lab_Gvec = np.array([1,0,0]).T
    #______________________________________________________________________
    # Compute the rotation matrix
    C = np.cross(lab_Gvec,iGvec)
    D = np.dot(lab_Gvec,iGvec)
    Z = np.array([[0,-C[2],C[1]],[C[2],0,-C[0]],[-C[1],C[0],0]])
    # Note: test the rotation matrix (R) that the norm of the colums and the rows is 1
    R = np.eye(3)+Z+(Z@Z)*(1-D)/np.linalg.norm(C)**2
    # Compute the rotation angles
    theta = np.zeros(3) # Preallocate
    theta[1] = arctan2(-R[2,0],np.sqrt(R[0,0]**2+R[1,0]**2))
    # Conditional statement in case theta[1] is +/- 90 deg (pi/2 rad)
    if theta[1] == np.pi/2:
        theta[2] = 0
        theta[0] = arctan2(R[0,1],R[1,1])
    elif theta[1] == -np.pi/2:
        theta[2] = 0
        theta[0] = -arctan2(R[0,1],R[1,1])
    else: 
        theta[2] = arctan2(R[1,0]/cos(theta[1]),R[0,0]/cos(theta[1]))
        theta[0] = arctan2(R[2,1]/cos(theta[1]),R[2,2]/cos(theta[1]))
    theta = theta*180/np.pi
    return theta

def rot2gravity(timeseries, HS, MS, GS):
    """
    Function to find the rotation of the foot flat acceleration vector to the 
    defined lab gravity coordinate system.

    Parameters
    ----------
    accel_vec : numpy array
        3x1 vector of the x,y,z acceleration at foot flat

    Returns
    -------
    theta : numpy array
        3x1 vector for the rotation from the foot flat acceleration to the lab
        coordinate system

    """
    #timeseries = up
    
    MS_frames = 20
    
    GS = GS[0:-2]
    
    
    for count,jj in enumerate(GS[:-1]):
        # Obtain the acceleration and gyroscope from foot contact (HS) to the
        # subsiquent midstance (MS). MS is needed for linear drift correction
        
        #count = 0
        #jj = GS[count]
        
        jjn = GS[count +1]
        acc_stride_plus = timeseries[HS[jj]:MS[jjn],4:7]
        gyr_stride_plus = timeseries[HS[jj]:MS[jjn], 1:4]
        time_stride_plus = timeseries[HS[jj]:MS[jjn],1]
        
        
        # Need the midstance and next foot contact indices
        MS_idx = MS[jj]-HS[jj]
        HS_idx = HS[jjn]-HS[jj]
        
        Fflat_accel = np.mean(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 0)/np.mean(np.linalg.norm(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 1)).T
        
        # Rotation to the gravity vector: provides initial contidion for gyro integration
        thetai = findRotToLab(Fflat_accel)
        
        # Integrate and linearly correct the gyroscope
        theta = cumtrapz(gyr_stride_plus,time_stride_plus,initial=0,axis=0)
        theta = theta-theta[MS_idx,:] # Midstance should be zero
        # Linear correction
        slope = theta[-1,:]/(len(theta)-MS_idx-1)
        drift = (slope*np.ones([len(theta),3]))*(np.array([range(len(theta)),range(len(theta)),range(len(theta))])).T
        drift = drift-drift[MS_idx,:]
        # Gyro integration initial condition
        if count == 0:
           theta = theta-thetai
           thetai_up = thetai
        else: 
           theta = theta-(thetai+thetai_up)/2
           thetai_up = thetai
        
        # Convert to radians
        theta = theta*np.pi/180
        # Rotate the accelerometer into the lab coordinate system (LCS)
        
        # Note: this rotation has been converted to list comprehension for speed purposes
        acc_stride_plus_LCS = [(np.array([[cos(ang[2]),-sin(ang[2]),0],[sin(ang[2]),cos(ang[2]),0],[0,0,1]])@np.array([[1,0,0],[0,cos(ang[0]),-sin(ang[0])],[0,sin(ang[0]),cos(ang[0])]])@np.array([[cos(ang[1]),0,sin(ang[1])],[0,1,0],[-np.sin(ang[1]),0,cos(ang[1])]])@acc_stride_plus[kk,:].T).T-np.array([9.81,0,0]) for kk,ang in enumerate(theta)]
        acc_stride_plus_LCS = np.stack(acc_stride_plus_LCS)
        acc_stride_plus_LCS = pd.DataFrame(acc_stride_plus_LCS[:HS_idx,])
        
        timeseries[HS[jj]:HS[jjn],1:4] = acc_stride_plus_LCS
        
        #rotatedAccDat.append(acc_stride_plus_LCS)
    
    
        # The written out loop for the rotation is as follows:
        # acc_stride_plus_LCS = np.zeros([len(acc_stride_plus),3])
        # for kk,ang in enumerate(theta):
        #     # Create the rotation matrix
        #     rotZ = np.array([[cos(ang[2]),-sin(ang[2]),0],
        #                     [sin(ang[2]),cos(ang[2]),0],
        #                     [0,0,1]])
        #     rotY = np.array([[cos(ang[1]),0,sin(ang[1])],
        #                     [0,1,0],
        #                     [-np.sin(ang[1]),0,cos(ang[1])]])
        #     rotX = np.array([[1,0,0],
        #                     [0,cos(ang[0]),-sin(ang[0])],
        #                     [0,sin(ang[0]),cos(ang[0])]])
            
        #     rot = rotZ @ rotX @ rotY
        #     acc_stride_plus_LCS[kk,:] = (rot @ acc_stride_plus[kk,:].T).T-np.array([9.81,0,0])
    
    
    return timeseries



### Import mocap data (for example)


fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/EndurancePerformance/TrailRun_2022/OutdoorData/IMUData/'

GPStiming = pd.read_csv('C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/EndurancePerformance/TrailRun_2022/OutdoorData/CombinedGPS.csv')

# Hentries_pelvis = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03399') ]
# Lentries_pelvis = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03399')]

Hentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('00218') ]
Lentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('00218')]



badFileList = []

for ii, fName in enumerate(Lentries_foot):
    
    try:
        #ii = 0
        #fName = Lentries_foot[ii]
        
        
        sub = fName.split('-')[0]
        conf = fName.split('-')[1]
        trial = fName.split('-')[2].split('_')[0]
        
       
        # Load the trials here
        # Ldf_pel = pd.read_csv(fPath + Lentries_pelvis[ii],sep=',', header = 0)
        # Hdf_pel = pd.read_csv(fPath + Hentries_pelvis[ii],sep=',', header = 0)
        
        Ldf_foot = pd.read_csv(fPath + Lentries_foot[ii],sep=',', header = 0)
        Hdf_foot = pd.read_csv(fPath + Hentries_foot[ii],sep=',', header = 0)
        
        [IMUtime_foot, iacc_foot, igyr_foot] = align_fuse_extract_IMU(Ldf_foot, Hdf_foot)
        # [IMUtime_pel,iacc_pel,igyr_pel] = align_fuse_extract_IMU(Ldf_pel,Hdf_pel)
        
        # idx = (IMUtime_foot < np.max(IMUtime_pel))*(IMUtime_foot > np.min(IMUtime_pel))
        # IMUtime_foot = IMUtime_foot[idx]
        # iacc_foot = iacc_foot[idx,:]
        # igyr_foot = igyr_foot[idx,:]
        
        #resamp_iacc_pel = np.zeros([len(IMUtime_foot),3])
        #resamp_igyr_pel = np.zeros([len(IMUtime_foot),3])
        
        # for jj in range(3):
        #     f = scipy.interpolate.interp1d(IMUtime_pel,iacc_pel[:,jj])
        #     resamp_iacc_pel[:,jj] = f(IMUtime_foot)
        #     f = scipy.interpolate.interp1d(IMUtime_pel,igyr_pel[:,jj])
        #     resamp_igyr_pel[:,jj] = f(IMUtime_foot)
        
        # Convert the time
        IMUtime = (IMUtime_foot - IMUtime_foot[0])*(1e-6)        
        # Identify foot contact & midstance
        
                      
        ### Parse data into uphill, flat and downhill sections
        
        tmpGPS = GPStiming[(GPStiming.Subject == sub) & (GPStiming.Config == conf)].reset_index()
        
        upGyr_foot = igyr_foot[(IMUtime > 30) & (IMUtime<tmpGPS.EndS1[0])]
        upAccel_foot = iacc_foot[(IMUtime > 30) & (IMUtime<tmpGPS.EndS1[0])]
        #upAccel_pel = resamp_iacc_pel[(IMUtime>30) & (IMUtime<tmpGPS.EndS1[0])]
        upTime = IMUtime[(IMUtime > 30) & (IMUtime<tmpGPS.EndS1[0])]
        up = np.concatenate(( upTime[:, None], upGyr_foot, upAccel_foot), axis = 1)
        
        
        flatGyr_foot = igyr_foot[(IMUtime > tmpGPS.StartS2[0]) & (IMUtime < tmpGPS.EndS2[0])]
        flatAccel_foot = iacc_foot[(IMUtime > tmpGPS.StartS2[0]) & (IMUtime < tmpGPS.EndS2[0])]
        #flatAccel_pel = resamp_iacc_pel[(IMUtime > tmpGPS.StartS2[0]) & (IMUtime < tmpGPS.EndS2[0])]
        flatTime = IMUtime[(IMUtime > tmpGPS.StartS2[0]) & (IMUtime < tmpGPS.EndS2[0])]
        flat = np.concatenate((flatTime[:, None], flatGyr_foot, flatAccel_foot), axis = 1)
       
        
        downGyr_foot = igyr_foot[(IMUtime> tmpGPS.StartS3[0]) & (IMUtime < (tmpGPS.StartS3[0] + tmpGPS.TimeS3[0]))]
        downAccel_foot = iacc_foot[(IMUtime> tmpGPS.StartS3[0]) & (IMUtime < (tmpGPS.StartS3[0] + tmpGPS.TimeS3[0]))]
        #downAccel_pel = resamp_iacc_pel[(IMUtime> tmpGPS.StartS3[0]) & (IMUtime < (tmpGPS.StartS3[0] + tmpGPS.TimeS3[0]))]
        downTime = IMUtime[(IMUtime> tmpGPS.StartS3[0]) & (IMUtime < (tmpGPS.StartS3[0] + tmpGPS.TimeS3[0]))]
        down = np.concatenate((downTime[:, None], downGyr_foot, downAccel_foot), axis = 1)
        
        
               
        for section in [up, flat, down]:
            
            #section = up
            
            
            sampEntX_foot = []
            sampEntY_foot = []
            sampEntZ_foot = []
            
            
                       
            [HS, MS] = estIMU_HS_MS(section[:,4:], section[:,1:4], section[:,0], 1e8)
            HS_t = section[HS, 0]
            GS = np.where((np.diff(HS) > 0.5)*(np.diff(HS_t) < 1.5))[0]
            
            section = rot2gravity(section, HS, MS, GS)
             
            dat20 = section[HS[0]:HS[20]+1, :]
            
            #plt.figure()
            #plt.plot(dat20[:, 4])
            
            sampEntX_foot.append(nolds.sampen(dat20[:, 4], emb_dim = 3, tolerance = 0.2, ))
            sampEntY_foot.append(nolds.sampen(dat20[:, 5], emb_dim = 3, tolerance = 0.2, ))
            sampEntZ_foot.append(nolds.sampen(dat20[:, 6], emb_dim = 3, tolerance = 0.2, ))
            # sampEntX_pel.append(nolds.sampen(dat20[:, 7], emb_dim = 3, tolerance = 0.2, ))
            # sampEntY_pel.append(nolds.sampen(dat20[:, 8], emb_dim = 3, tolerance = 0.2, ))
            # sampEntZ_pel.append(nolds.sampen(dat20[:, 9], emb_dim = 3, tolerance = 0.2, ))
            
           
                      
            if np.array_equiv(section, up) :
                slp = 'up'
            elif np.array_equiv(section, flat):
                slp = 'flat'
            else:
                slp = 'down'
            
                  
            outcomes = pd.DataFrame({'Subject':sub, 'Config':conf, 'Slope':slp, 'Trial':trial, 
                                     'sampEntX_foot':list(sampEntX_foot), 'sampEntY_foot':list(sampEntY_foot), 'sampEntZ_foot':list(sampEntZ_foot)})
            outfileName = fPath + 'CompiledSampleEntropy_Rotated.csv'
            if os.path.exists(outfileName) == False:
                
                outcomes.to_csv(outfileName, header=True, index = False)
        
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
        print(str(ii+1) + ' / ' + str(len(Lentries_foot)) + ' Good')
             
    except:
        badFileList.append(fName)
        print(str(ii +1) + ' / ' + str(len(Lentries_foot)) + ' Bad')
        
