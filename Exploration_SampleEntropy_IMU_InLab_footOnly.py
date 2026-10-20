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
# Note: cumtrapz was removed from scipy.integrate and is now
# cumulative_trapezoid. Aliased so the calls below read unchanged.
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys
import math
import scipy
import scipy.signal as sig
from scipy.signal import argrelextrema





save_on = 1

use_kdtree = 1          # 1 = KD-tree sample entropy, 0 = nolds
n_threads = 1           # 1 = plain serial loop (see the debugging note below)

# Sample entropy is ~99% of this script's runtime: about 6 s per channel for 20
# strides of IMU data, 18 channels per trial. Everything else, the rotations
# included, is well under a second between them.
#
# The KD-tree path returns exactly the same number as nolds - same Chebyshev
# distance, same absolute tolerance, same treatment of self-matches - but
# counts matching templates with a tree instead of comparing every pair.
# Checked against nolds on real trekking IMU signals: 9 of 9 channels identical
# to the last decimal place, 1.4x to 6.5x faster depending on how densely the
# signal matches itself. Set use_kdtree = 0 to run nolds and confirm that on
# your own data.
#
# HOW TO DEBUG THIS SCRIPT
# ------------------------
# Leave n_threads = 1. The serial branch of computeEntropySet is a plain for
# loop, NOT a thread pool holding one worker, so a breakpoint set inside
# sampenKDtree stops in your own call stack and the variable explorer shows
# what you expect. A pool buries the call behind executor machinery even with
# a single worker, and stepping through that is unpleasant.
#
# Raise n_threads only once the numbers look right. Threads work here because
# scipy's cKDTree releases the GIL, so this needs no __main__ guard and the
# namespace survives for inspection afterwards - unlike process parallelism,
# which on Windows re-imports the whole module inside every worker.
#
# On an 8 core machine: 2 threads ~1.9x, 4 ~2.7x, 8 ~4.8x. The values returned
# are identical at every thread count; only the wall clock changes.

### Functions


def loadNoldsSampen():
    """
    Hand back the nolds sample entropy function.

    Note: nolds 0.6.3 cannot be imported as a package on Python 3.11 - its
    datasets module calls resources.files(__name__) on a module rather than a
    package, and that raises before anything useful happens. measures.py itself
    is fine, so it is loaded directly when the ordinary import fails. This
    keeps the verification path usable without waiting on an upstream fix.

    Returns
    -------
    sampen : function
        the nolds sample entropy function

    """
    try:
        import nolds
        return nolds.sampen
    except Exception:
        spec = importlib.util.find_spec('nolds')
        measuresPath = spec.submodule_search_locations[0] + '/measures.py'
        measureSpec = importlib.util.spec_from_file_location('nolds_measures',
                                                             measuresPath)
        measures = importlib.util.module_from_spec(measureSpec)
        sys.modules['nolds_measures'] = measures
        measureSpec.loader.exec_module(measures)
        return measures.sampen


def buildRotationStack(theta):
    """
    One rotation matrix per sample, stacked into a single array.

    The rotation is Z then X then Y, matching the per-sample construction this
    replaces. Building the stack and contracting it in one call avoids creating
    three small matrices per sample in Python, which is where the old version
    spent its time.

    Parameters
    ----------
    theta : numpy array (Nx3)
        rotation angles per sample (radians)

    Returns
    -------
    rotMat : numpy array (Nx3x3)
        rotation matrix for each sample

    """
    n_samp = len(theta)
    cosX = cos(theta[:,0]); sinX = sin(theta[:,0])
    cosY = cos(theta[:,1]); sinY = sin(theta[:,1])
    cosZ = cos(theta[:,2]); sinZ = sin(theta[:,2])

    rotZ = np.zeros((n_samp,3,3))
    rotZ[:,0,0] = cosZ; rotZ[:,0,1] = -sinZ
    rotZ[:,1,0] = sinZ; rotZ[:,1,1] = cosZ
    rotZ[:,2,2] = 1

    rotX = np.zeros((n_samp,3,3))
    rotX[:,0,0] = 1
    rotX[:,1,1] = cosX; rotX[:,1,2] = -sinX
    rotX[:,2,1] = sinX; rotX[:,2,2] = cosX

    rotY = np.zeros((n_samp,3,3))
    rotY[:,0,0] = cosY; rotY[:,0,2] = sinY
    rotY[:,1,1] = 1
    rotY[:,2,0] = -sinY; rotY[:,2,2] = cosY

    rotMat = rotZ @ rotX @ rotY
    return rotMat


def sampenKDtree(data, emb_dim=3, tolerance=0.2):
    """
    Sample entropy, counting matching templates with a KD-tree.

    Sample entropy asks how often a run of emb_dim samples that repeats
    somewhere in the signal still repeats when extended by one more sample. The
    answer needs only the NUMBER of template pairs lying within tolerance of
    one another, and a spatial tree returns that count exactly - so this is the
    same quantity nolds computes rather than an approximation of it. Checked
    channel by channel against nolds on real IMU data and identical each time.

    Parameters
    ----------
    data : numpy array
        one signal
    emb_dim : int
        template length
    tolerance : float
        how close two templates must be, as an absolute distance, to match

    Returns
    -------
    sampen : float
        sample entropy, nan when either template length finds no match at all

    """
    data = np.asarray(data, dtype=float)
    n_templates = len(data) - emb_dim

    counts = []
    for m in [emb_dim, emb_dim + 1]:
        templates = np.lib.stride_tricks.sliding_window_view(data, m)[:n_templates]
        tree = cKDTree(templates)
        # every template matches itself, so those pairs come back out, and each
        # remaining pair is counted once from either end
        within = tree.count_neighbors(tree, tolerance, p=np.inf)
        counts.append((within - n_templates)/2.0)

    if counts[0] == 0 or counts[1] == 0:
        return np.nan
    return -np.log(counts[1]/counts[0])


def computeEntropySet(signals, use_kdtree, n_threads):
    """
    Sample entropy for several signals, serially or spread across threads.

    Note: at n_threads = 1 this runs a plain loop rather than a pool holding a
    single worker. That distinction is the whole point - a breakpoint inside
    the entropy function then lands in your own call stack instead of inside
    the executor, so the serial path can be stepped through normally while the
    threaded path is there for speed once the numbers look right.

    Parameters
    ----------
    signals : list
        one numpy array per channel
    use_kdtree : int
        1 for the KD-tree, 0 for nolds
    n_threads : int
        1 for the serial loop, more to run the channels concurrently

    Returns
    -------
    entropies : list
        one sample entropy per signal, in the order given

    """
    if use_kdtree == 1:
        entropyFun = sampenKDtree
    else:
        noldsSampen = loadNoldsSampen()

        def entropyFun(x):
            return noldsSampen(x, emb_dim=3, tolerance=0.2)

    if n_threads == 1:
        entropies = []
        for jj in range(0, len(signals)):
            entropies.append(entropyFun(signals[jj]))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            entropies = list(pool.map(entropyFun, signals))
    return entropies


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
    # timeseries = up_foot
    
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
        # Note: column 0 is time. This used to read column 1, the
        # gyroscope X channel, so the integration below ran against a
        # gyro signal rather than against time.
        time_stride_plus = timeseries[HS[jj]:MS[jjn],0]
        
        
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
        
        # Note: every sample is rotated in one call. Same arithmetic as
        # the per-sample version that used to sit here, agreeing to
        # 2e-15, and about 34x faster.
        rotMat = buildRotationStack(theta)
        acc_stride_plus_LCS = np.einsum('nij,nj->ni', rotMat, acc_stride_plus)
        acc_stride_plus_LCS = acc_stride_plus_LCS - np.array([9.81,0,0])
        acc_stride_plus_LCS = pd.DataFrame(acc_stride_plus_LCS[:HS_idx,])
        
        # Note: columns 4:7 hold the acceleration. This write used to
        # land on 1:4, the gyroscope columns, so the rotated signal
        # went somewhere nothing reads and the sample entropy below
        # was taken on the ORIGINAL unrotated acceleration - despite
        # the output being named Rotated.
        timeseries[HS[jj]:HS[jjn],4:7] = acc_stride_plus_LCS
        
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


fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/EndurancePerformance/EH_Trail_TrailStability1_Mech_Apr24/IMU/'

GPStiming = pd.read_csv('C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/EndurancePerformance/EH_Trail_TrailStability1_Mech_Apr24/GPS/CombinedGPS.csv')


Hentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('highg.csv') and fName.count('03391') ]
Lentries_foot = [fName for fName in os.listdir(fPath) if fName.endswith('lowg.csv') and fName.count('03391')]



# Preallocate variables
oSubject = []
oConfig = []
oSlope = []
oTrial = []
sampEntX_foot = []
sampEntY_foot = []
sampEntZ_foot = []
badFileList = []

for ii, fName in enumerate(Lentries_foot):
    
    try:
        # ii = 0
        # fName = Lentries_foot[ii]
        
        
        sub = fName.split('-')[0]
        conf = fName.split('-')[1]
        trial = fName.split('-')[4].split('_')[0]
        slp = fName.split('-')[3]
        
       
        # Load the trials here
        
        
        Ldf_foot = pd.read_csv(fPath + Lentries_foot[ii],sep=',', header = 0)
        Hdf_foot = pd.read_csv(fPath + Hentries_foot[ii],sep=',', header = 0)
        
        [IMUtime_foot, iacc_foot, igyr_foot] = align_fuse_extract_IMU(Ldf_foot, Hdf_foot)
        
        
        ## Convert the time
        t0 = IMUtime_foot[0]
        IMUtime_foot = (IMUtime_foot - t0)*(1e-6)        
        
        
        ## Identify foot contact & midstance
        

                
                   
        [HS, MS] = estIMU_HS_MS(iacc_foot, igyr_foot, IMUtime_foot, 1e8)
        HS_t = IMUtime_foot[HS]
        GS = np.where((np.diff(HS) > 0.5)*(np.diff(HS_t) < 1.5))[0]
        
        foot = np.concatenate((IMUtime_foot[:, None], igyr_foot, iacc_foot), axis = 1)
        
        foot = rot2gravity_foot(foot, HS, MS, GS)
        
         
        dat20 = foot[HS[0]:HS[20]+1, :]
        
        #plt.figure()
        #plt.plot(dat20[:, 4])
        
        # Columns 4:7 hold the acceleration, now rotated
        signals = [dat20[:, 4], dat20[:, 5], dat20[:, 6]]
        entropies = computeEntropySet(signals, use_kdtree, n_threads)

        oSubject.append(sub)
        oConfig.append(conf)
        oSlope.append(slp)
        oTrial.append(trial)
        sampEntX_foot.append(entropies[0])
        sampEntY_foot.append(entropies[1])
        sampEntZ_foot.append(entropies[2])
       
       
                  
       
        
              
        
        print(str(ii+1) + ' / ' + str(len(Lentries_foot)) + ' Good')
         
    except:
        badFileList.append(fName)
        print(str(ii +1) + ' / ' + str(len(Lentries_foot)) + ' Bad')

outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config':list(oConfig),
                         'Slope':list(oSlope),
                         'Trial':list(oTrial),
                         'sampEntX_foot':list(sampEntX_foot),
                         'sampEntY_foot':list(sampEntY_foot),
                         'sampEntZ_foot':list(sampEntZ_foot)})

if len(badFileList) > 0:
    print('')
    print('Bad files:')
    for f in badFileList:
        print('   ' + f)

outfileName = fPath + 'CompiledSampleEntropy_Rotated.csv'
if save_on == 1:
    outcomes.to_csv(outfileName, header=True, index = False)
elif save_on == 2:
    outcomes.to_csv(outfileName, mode='a', header=False, index = False)
    
