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
from scipy.integrate import cumtrapz
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import time
import addcopyfighandler

# Functions
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
        LGtime = LGtime[:-lag]
        gyr = gyr[:-lag,:]
        acc_lg = acc_lg[:-lag,:]
        resamp_HG = resamp_HG[lag:,:]
    
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

def computeRunSpeedIMU(acc,gyr,HS,MS,GS,t):
    """
    Estimate running speed from IMU signals. 
    
    This function rotates the accleration signal into an inertial coordinate
    system based on the integrand of the gyscrope signal. This rotation is
    performed in order to account for the gravitational acceleration. The
    corrected inertial coordinate system accleration is then double
    integrated to estimate the displacement. The integrations are linearly
    corrected. Note that all time continuous signals span from foot contact
    to the subsequent midstance. The final dispacement calculation is from
    foot contact to the subsiquent foot contact.

    Parameters
    ----------
    acc : numpy array (Nx3)
        X,Y,Z acceleration from the IMU
    gyr : numpy array (Nx3)
        X,Y,Z gyroscope from the IMU
    HS : list
        Heel-strike (foot contact events) indices
    MS : list
        Mid-stance indices
    GS : list
        Indices that indicate good strides
    t : numpy array (Nx1)
        time (seconds)

    Returns
    -------
    run_speed : list
        Step-by-step running speed

    """
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = 50 / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    
    # Filter the IMU signals
    acc_filt = np.array([sig.filtfilt(b, a, acc[:,jj]) for jj in range(3)]).T
    gyr_filt = np.array([sig.filtfilt(b, a, gyr[:,jj]) for jj in range(3)]).T    
    
    # Indicate the number of frames to examine the gravity during midstance
    MS_frames = 20
    
    run_speed = []
    
    for count,jj in enumerate(GS):
        # Obtain the acceleration and gyroscope from foot contact (HS) to the
        # subsiquent midstance (MS). MS is needed for linear drift correction
        acc_stride_plus = acc_filt[HS[jj]:MS[jj+1],:]
        gyr_stride_plus = gyr_filt[HS[jj]:MS[jj+1],:]
        time_stride_plus = t[HS[jj]:MS[jj+1]]
        time_stride = t[HS[jj]:HS[jj+1]]
        
        # Need the midstance and next foot contact indices
        MS_idx = MS[jj]-HS[jj]
        HS_idx = HS[jj+1]-HS[jj]
        
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

        # Integrate to get IMU velocity in the LCS
        IMU_vel = cumtrapz(acc_stride_plus_LCS,time_stride_plus,initial=0,axis=0)
        # Inital contidion: IMU velocity is zero at midstance
        IMU_vel = IMU_vel-IMU_vel[MS_idx,:]
        # Remove drift
        slope = IMU_vel[-1,:]/(len(IMU_vel)-MS_idx-1)
        drift = (slope*np.ones([len(IMU_vel),3]))*(np.array([range(len(IMU_vel)),range(len(theta)),range(len(IMU_vel))])).T
        drift = drift-drift[MS_idx,:]
        IMU_vel = IMU_vel-drift
        # Integrate to get step length in the LCS
        IMU_SL = np.trapz(IMU_vel[0:HS_idx,:],time_stride,axis=0)
        run_speed.append(np.linalg.norm(IMU_SL)/(time_stride[-1]-time_stride[0])) 
        
    return run_speed

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

def filtIMUsig(sig_in,cut,t):
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = np.array([sig.filtfilt(b, a, sig_in[:,jj]) for jj in range(3)]).T    
    return(sig_out)


# Obtain IMU signals
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\Hike_DualDialMechanistic_Aug22\\Pilot\\IMUdata\\'

save_on = 0

# High and Low G accelerometers: note that the gyro is in the low G file
Hentries = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv')]
Lentries = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv')]

# 
oSubject = []
oConfig = []
oSesh = []
oSpeed = []
oLabel = []
setting = []
oSide = []

pGyr = []
pAcc = []
pJerk = []
rMLacc = []
rIEgyro = []
imuSpeed = []

# Filtering frequencies
acc_cut = 50
gyr_cut = 30

# Index through the low-g files
for ii in range(2,len(Lentries)):
    print(Lentries[ii])
    # Load the trials here
    Ldf = pd.read_csv(fPath + Lentries[ii],sep=',', header = 0)
    Hdf = pd.read_csv(fPath + Hentries[ii],sep=',', header = 0)
    # Save trial information
    # Subject = Lentries[ii].split(sep = "-")[0]
    # Config = Lentries[ii].split(sep="-")[1]
    # Speed = Lentries[ii].split(sep="-")[2]
    # Slope = Lentries[ii].split(sep="-")[3]
    # Sesh = Lentries[ii].split(sep="-")[4][0]
    

    
    [IMUtime,iacc,igyr] = align_fuse_extract_IMU(Ldf,Hdf)
    # Convert the time
    IMUtime = (IMUtime - IMUtime[0])*(1e-6)        
    # Identify foot contact & midstance
    [iHS,iMS] = estIMU_HS_MS(iacc,igyr,IMUtime,8e8)
    
    # Examine where the start of the trial is by the 3 jumps
    # There should seem to be 2 "short" strides followed by a pause
    approx_CT = np.diff(iHS)
    iHS_t = IMUtime[iHS]
    
    # Counters
    jc = 0  # jump counter
    stc = 0 # start trial counter
    jj = 0
    up_thresh = 8e8
    
    
    # Algorithm to detect 3 hops - will need to be updated
    # while stc == 0:
    #     if approx_CT[jj] < 1500:
    #         jc = jc+1
    #     if jc >= 2 and approx_CT[jj] > 2000:
    #         idx = (iHS_t > (iHS_t[jj] + 10))*(iHS_t < (iHS_t[jj] + 55))
    #         iHS = iHS[idx]
    #         iHS_t = iHS_t[idx]
    #         iMS = iMS[idx]
    #         stc = 1
        
    #     jj = jj+1
        
    #     if jj > 10:
    #         up_thresh = up_thresh - 2e8
    #         [iHS,iMS] = estIMU_HS_MS(iacc,igyr,IMUtime,up_thresh)
    #         approx_CT = np.diff(iHS)
    #         iHS_t = IMUtime[iHS]
    #         jj = 0
        
    
    plt.figure(ii)
    plt.plot(iacc[:,0])
    plt.plot(iHS,iacc[iHS,0],'ko')
    
    iGS = np.where((np.diff(iHS_t) > 0.5)*(np.diff(iHS_t) < 1.5))[0]
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
        
    # Appending
    # oSubject = oSubject + [Subject]*len(iGS)
    # oConfig = oConfig + [Config]*len(iGS)
    # oLabel = oLabel + [Label]*len(iGS)
    # setting = setting + ['0']*len(iGS)
    # oSesh = oSesh + [Sesh]*len(iGS)
    # if Slope[0] == 'n':
    #     oSide = oSide + ['L']*len(iGS)
    # else: 
    #     oSide = oSide + ['R']*len(iGS)
    # Clear variables
    iHS = []; iGS = []
    
outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig), 'Movement':list(oLabel),
                         'Sesh': list(oSesh), 'pJerk':list(pJerk),'pAcc':list(pAcc), 'pGyr':list(pGyr),
                           'rMLacc':list(rMLacc),'rIEgyro':list(rIEgyro),'imuSpeed':list(imuSpeed)})


if save_on == 1:
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\CompIMUmetrics.csv',mode = 'a',header=False)


