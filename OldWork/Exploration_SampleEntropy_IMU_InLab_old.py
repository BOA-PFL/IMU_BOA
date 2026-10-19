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




def rot2gravity_foot(timeseries, HS, MS, GS):
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
    #timeseries = foot
    
    MS_frames = 20
    
    GS = GS[0:-2]
    
    
    for count,jj in enumerate(GS[:-1]):
        # Obtain the acceleration and gyroscope from foot contact (HS) to the
        # subsiquent midstance (MS). MS is needed for linear drift correction
        
        # count = 0
        # jj = GS[count]
        
        jjn = GS[count +1]
        acc_stride_plus = timeseries[HS[jj]:MS[jjn],4:7]
        gyr_stride_plus = timeseries[HS[jj]:MS[jjn], 1:4]
        time_stride_plus = timeseries[HS[jj]:MS[jjn],1]
        
        
        # Need the midstance and next foot contact indices
        MS_idx = MS[jj]-HS[jj]
        HS_idx = HS[jjn]-HS[jj]
        
        Fflat_accel = np.mean(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 0)/np.mean(np.linalg.norm(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 1)).T ### for pelvis change this to mean accel over whole trial
        
        # Rotation to the gravity vector: provides initial contidion for gyro integration
        thetai = findRotToLab(Fflat_accel)
        
        # Integrate and linearly correct the gyroscope
        theta = cumtrapz(gyr_stride_plus,time_stride_plus,initial=0,axis=0)
        theta = theta-theta[MS_idx,:] # Midstance should be zero                                                                                                  ### for pelvis change this to theta - theta(mean angle of whole trial)
        # Linear correction
        slope = theta[-1,:]/(len(theta)-MS_idx-1)
        drift = (slope*np.ones([len(theta),3]))*(np.array([range(len(theta)),range(len(theta)),range(len(theta))])).T
        drift = drift-drift[MS_idx,:]
        theta = theta - drift
        
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


def rot2gravity_pel(timeseries):
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
    #timeseries = pelvis
    
      
    
    meanAccel = np.mean(timeseries[:,4:7], axis = 0)/np.mean(np.linalg.norm(timeseries[:, 4:7], axis = 1)).T # Mean acceleration over whole trial gives mean orientation (assuming the subject starts and finishes the trial with the same acceleration - we do this by having them stand still when we start and stop recording) 
    
    
    # Rotation to the gravity vector: provides average pelvis orientation through trial 
    thetai = findRotToLab(meanAccel) ### Do we need to divide by the norm?
        
        # Integrate and linearly correct the gyroscope
    theta = cumtrapz(timeseries[:, 1:4],timeseries[:, 0],initial=0,axis=0) # This gives the DIFFERENCE from average orientation of the pelvis IMU
                                                                                                      
    # Linear correction
    slope = (theta[-1,:]-theta[0, :])/(len(theta)) # Subject starts and finishes trial standing in the same position. So the difference between first and last angle is the drift. 
    drift = (slope*np.ones([len(theta),3]))*(np.array([range(len(theta)),range(len(theta)),range(len(theta))])).T
    theta = theta - drift
        
    theta = theta + thetai # Subtract the offset of the IMU position from the gravity vector, calculated above
        
    # Convert to radians
    theta = theta*np.pi/180
    
    # Rotate the accelerometer into the lab coordinate system (LCS)
    # Note: this rotation has been converted to list comprehension for speed purposes
    newacc = [(np.array([[cos(ang[2]),-sin(ang[2]),0],[sin(ang[2]),cos(ang[2]),0],[0,0,1]])@np.array([[1,0,0],[0,cos(ang[0]),-sin(ang[0])],[0,sin(ang[0]),cos(ang[0])]])@np.array([[cos(ang[1]),0,sin(ang[1])],[0,1,0],[-np.sin(ang[1]),0,cos(ang[1])]])@timeseries[kk,4:7].T).T-np.array([9.81,0,0]) for kk,ang in enumerate(theta)]
    newacc = np.stack(newacc)
    newacc = pd.DataFrame(newacc)
    
    timeseries[:,4:7] = newacc
    
    
        # The written out loop for the rotation is as follows:
        # newacc = np.zeros([len(timeseries),3])
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
        #     newacc[kk,:] = (rot @ timeseris[kk,4:7].T).T-np.array([9.81,0,0])
    
    
    return timeseries


def intp_strides(var,landings):
    """
    Function to interpolate the variable of interest across a stride
    (from foot contact to subsiquent foot contact) in order to plot the 
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    landings : list
        Foot contact indicies
    takeoffs : list
        Toe-off indicies

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(landings)))
    # Index through the strides
    for ii in range(len(landings)-1):
        dum = var[landings[ii]:landings[ii+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var
### Import mocap data (for example)


fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/WorkWear_Performance/EH_Workwear_PFSLiteI_Perf_Aug24/IMU/'



Hentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03391')]
Lentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03391')]

# Hentries_pel = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03399') ]
# Lentries_pel = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03399')]



badFileList = []

for ii, fName in enumerate(Lentries_foot):
    
    try:
        #ii = 6
        #fName = Lentries_foot[ii]
        
        
        sub = fName.split('-')[1]
        conf = fName.split('-')[2]
        trial = fName.split('-')[4].split('_')[0]
        mvmt = fName.split('-')[3]
        
        if 'rail' in mvmt:
        # Load the trials here
        
        
            Ldf_foot = pd.read_csv(fPath + Lentries_foot[ii],sep=',', header = 0)
            Hdf_foot = pd.read_csv(fPath + Hentries_foot[ii],sep=',', header = 0)
            
            # Ldf_pel = pd.read_csv(fPath + Lentries_pel[ii],sep=',', header = 0)
            # Hdf_pel = pd.read_csv(fPath + Hentries_pel[ii],sep=',', header = 0)
            
            [IMUtime_foot, iacc_foot, igyr_foot] = align_fuse_extract_IMU(Ldf_foot, Hdf_foot)
            # [IMUtime_pel, iacc_pel, igyr_pel] = align_fuse_extract_IMU(Ldf_pel, Hdf_pel)
            
            
            ## Convert the time
            t0 = IMUtime_foot[0]
            IMUtime_foot = (IMUtime_foot - t0)*(1e-6)        
            # IMUtime_pel = (IMUtime_pel - t0)*(1e-6)
            
            # if IMUtime_pel[0] > 0:
            #     tdiff = abs(IMUtime_foot - IMUtime_pel[0])
            #     minidx = np.argmin(tdiff)
            #     IMUtime_foot = IMUtime_foot[minidx:]
            #     iacc_foot = iacc_foot[minidx:, :]
            #     igyr_foot = igyr_foot[minidx:, :]
                
            # else:
            #     minidx = np.argmin(abs(IMUtime_pel))
            #     IMUtime_pel = IMUtime_pel[minidx:]
            #     iacc_pel = iacc_pel[minidx:, :]
            #     igyr_pel = igyr_pel[minidx:, :]
                
                   
            ## Identify foot contact & midstance
            
    
            sampEntX_foot = []
            sampEntY_foot = []
            sampEntZ_foot = []
            
            # sampEntX_pel = []
            # sampEntY_pel = []
            # sampEntZ_pel = []
                    
                       
            [HS, MS] = estIMU_HS_MS(iacc_foot, igyr_foot, IMUtime_foot, 1e8)
            HS_t = IMUtime_foot[HS]
            GS = np.where((np.diff(HS) > 0.5)*(np.diff(HS_t) < 1.5))[0]
            
            foot = np.concatenate((IMUtime_foot[:, None], igyr_foot, iacc_foot), axis = 1)
            # pelvis = np.concatenate((IMUtime_pel[:, None], igyr_pel, iacc_pel), axis = 1)
            foot = rot2gravity_foot(foot, HS, MS, GS)
            
            
            # meanstride = np.mean(intp_strides(foot[:, 4], HS), axis = 1)
            # pelvis = rot2gravity_pel (pelvis)
            
             
            dat20_foot = foot[HS[0]:HS[20]+1, :]
            
            #plt.figure()
            #plt.plot(dat20[:, 4])
            
            sampEntX_foot.append(nolds.sampen(dat20_foot[:, 4], emb_dim = 3, tolerance = 0.2, ))
            sampEntY_foot.append(nolds.sampen(dat20_foot[:, 5], emb_dim = 3, tolerance = 0.2, ))
            sampEntZ_foot.append(nolds.sampen(dat20_foot[:, 6], emb_dim = 3, tolerance = 0.2, ))
           
            
            # dat20_pel = pelvis[HS[0]:HS[20]+1, :]
            
            # #plt.figure()
            # #plt.plot(dat20[:, 4])
            
            # sampEntX_pel.append(nolds.sampen(dat20_pel[:, 4], emb_dim = 3, tolerance = 0.2, ))
            # sampEntY_pel.append(nolds.sampen(dat20_pel[:, 5], emb_dim = 3, tolerance = 0.2, ))
            # sampEntZ_pel.append(nolds.sampen(dat20_pel[:, 6], emb_dim = 3, tolerance = 0.2, ))
                   
           
            
                  
            outcomes = pd.DataFrame({'Subject':sub, 'Config':conf, 'Trial':trial, 
                                     'sampEntX_foot':list(sampEntX_foot), 'sampEntY_foot':list(sampEntY_foot), 'sampEntZ_foot':list(sampEntZ_foot)
                                     #'sampEntX_pel':list(sampEntX_pel), 'sampEntY_pel':list(sampEntY_pel), 'sampEntZ_pel':list(sampEntZ_pel)
                                     })
            outfileName = fPath + 'CompiledSampleEntropy_Rotated.csv'
            if os.path.exists(outfileName) == False:
                
                outcomes.to_csv(outfileName, header=True, index = False)
        
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
            print(str(ii+1) + ' / ' + str(len(Lentries_foot)) + ' Good')
         
    except:
        badFileList.append(fName)
        print(str(ii +1) + ' / ' + str(len(Lentries_foot)) + ' Bad')
    
