# -*- coding: utf-8 -*-
"""
Shared functions for IMU-based biomechanical analyses. These functions
are used across multiple IMU processing scripts.

@author: Eric.Honert
"""

import os
import numpy as np
from numpy import cos, sin, arctan2
import scipy
import scipy.interpolate
import scipy.signal as sig
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt


def filtIMUsig(sig_in, cut, t):
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

def delimitTrialIMU(SegSig):
    """
    Function to crop the data using interactive ginput selection

    Parameters
    ----------
    SegSig : numpy array or list
        The signal that will be plotted to allow for segmentation

    Returns
    -------
    segidx : array
        Segmentation indices (start and end)

    """
    print('Select 2 points: the start and end of the trial')
    fig, ax = plt.subplots()
    ax.plot(SegSig, label = 'Segmenting Signal')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=40))
    plt.close()
    segidx = pts[:,0]
    return(segidx)

def selectStudyFoot(FilePath):
    """
    Ask once which foot the IMU was worn on, and remember it for the study.

    A single-IMU indoor protocol puts the sensor on the same foot for every
    subject and every trial, so this is a property of the study rather than of
    a file. It is asked once, saved next to the data as StudyFoot.npy, and read
    back silently on every later run -- the same load-or-prompt pattern as
    delimitTrial in XSENSORFunctions.

    It matters because the inversion/eversion gyroscope is signed about the
    fore-aft axis and the two feet are mirror images, so a formula written for
    one foot reports eversion where the other reports inversion. Getting it
    wrong does not raise anything; it silently swaps two output columns.

    To change the answer later, delete StudyFoot.npy and re-run.

    Parameters
    ----------
    FilePath : str
        Folder holding the study's IMU files. The selection is saved here, so
        it travels with the data rather than with the script.

    Returns
    -------
    Foot : str
        'Left' or 'Right'.

    """
    savePath = FilePath + 'StudyFoot.npy'

    if os.path.exists(savePath) == True:
        Foot = str(np.load(savePath, allow_pickle=True))
        print('Study foot: %s (from StudyFoot.npy)' % Foot)
        return(Foot)

    # Imported here rather than at the top of the module: IMUFunctions is
    # imported by scripts that may run without a display, and a missing tk
    # should not stop them loading.
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    isLeft = messagebox.askyesno('Study foot',
                                 'Which foot was the IMU worn on for this study?'
                                 '\n\nYes  =  LEFT foot'
                                 '\nNo   =  RIGHT foot')
    root.destroy()

    if isLeft == True:
        Foot = 'Left'
    else:
        Foot = 'Right'

    np.save(savePath, np.array(Foot, dtype=object))
    print('Study foot: %s (saved to StudyFoot.npy)' % Foot)

    return(Foot)


def xcorr_fun(parent_sig, child_sig, tot_shift):
    """
    Windowed Pearson cross-correlation between two signals.

    Only the child signal is shifted, over a bounded range of lags. Two reasons
    this is preferred over maximising a raw correlation across every lag:

    * The value returned at each lag is a Pearson coefficient, so it reads on a
      fixed -1 to 1 scale. A raw correlation magnitude depends on signal
      amplitude and tells you nothing on its own about how good the match is,
      which makes a bad alignment hard to recognise when stepping through.
    * Bounding the search means a lag larger than physically plausible cannot be
      returned at all. Where two accelerometers in one unit are being aligned,
      the true offset is a sample or two, so a wide search only offers chances
      to pick up a spurious match.

    Parameters
    ----------
    parent_sig : array_like (N,)
        Reference signal, held fixed.
    child_sig : array_like (N,)
        Signal to be shifted.
    tot_shift : int
        Maximum shift in either direction; the search covers
        [-tot_shift, +tot_shift] samples.

    Returns
    -------
    xcorr_lag : numpy array (2*tot_shift+1,)
        Lag, in samples, for each correlation value.
    xcorr_corr : numpy array (2*tot_shift+1,)
        Pearson correlation coefficient at each lag.

    Notes
    -----
    The coefficient at every lag is assembled from one FFT cross-correlation
    (the sum of products) plus cumulative sums of the zero-padded child (its sum
    and sum of squares), rather than by rebuilding and correlating the signal at
    each lag in turn. On a 1.25M-sample file that is ~0.5 s against ~35 s for the
    loop, with results equal to within 8e-16.

    ONE CAVEAT. Variance here is computed as E[x^2] - E[x]^2, which loses
    precision when a signal carries a large DC offset relative to its variance.
    Measured against a mean-subtracting reference: ~1e-16 at zero mean, 7e-10 at
    mean 1e3 / SD 1, 2.8e-3 at mean 1e6 / SD 1. The returned LAG was correct in
    every case. Accelerometer axes are near zero mean so this does not arise in
    align_fuse_extract_IMU, but it WOULD matter for a strictly positive signal
    such as a GRF or a pressure trace -- subtract the mean of each signal before
    calling if you use this function for that.

    """
    parent_sig = np.asarray(parent_sig, dtype=float)
    child_sig = np.asarray(child_sig, dtype=float)

    # Zero-pad the shorter signal so both are equal length
    if len(parent_sig) < len(child_sig):
        parent_sig = np.concatenate((parent_sig,
                                     np.zeros(len(child_sig) - len(parent_sig))))
    elif len(parent_sig) > len(child_sig):
        child_sig = np.concatenate((child_sig,
                                    np.zeros(len(parent_sig) - len(child_sig))))

    N = len(parent_sig)
    xcorr_lag = np.arange(-tot_shift, tot_shift + 1, dtype=float)
    lags = xcorr_lag.astype(int)

    full = sig.correlate(parent_sig, child_sig, mode='full', method='fft')
    Spm = full[lags + (N - 1)]

    cs = np.concatenate(([0.0], np.cumsum(child_sig)))
    cs2 = np.concatenate(([0.0], np.cumsum(child_sig**2)))
    pos = np.clip(lags, 0, None)
    neg = np.clip(-lags, 0, None)
    # shift >= 0 keeps child[:N-shift]; shift < 0 keeps child[|shift|:]
    Sm = np.where(lags >= 0, cs[N - pos], cs[N] - cs[neg])
    Sm2 = np.where(lags >= 0, cs2[N - pos], cs2[N] - cs2[neg])

    p_mean = np.mean(parent_sig)
    p_var = np.mean(parent_sig**2) - p_mean**2
    m_mean = Sm/N
    m_var = Sm2/N - m_mean**2

    cov = Spm/N - p_mean*m_mean
    with np.errstate(invalid='ignore', divide='ignore'):
        xcorr_corr = cov/np.sqrt(p_var*m_var)

    # Guard against zero-variance signals, which would divide by zero
    bad = (np.sqrt(max(p_var, 0.0)) < 1e-12) | (np.sqrt(np.clip(m_var, 0, None)) < 1e-12)
    xcorr_corr[bad] = 0.0

    return xcorr_lag, xcorr_corr


def align_fuse_extract_IMU(LGdat, HGdat):
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
        high-g data frame that is extracted as raw data from Capture.U.

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

    idx = (LGtime < np.max(HGtime))*(LGtime >= np.min(HGtime))
    LGtime = LGtime[idx]
    acc_lg = acc_lg[idx,:]
    gyr = gyr[idx,:]

    # Create an empty array to fill with the resampled/downsampled high-g acceleration
    resamp_HG = np.zeros([len(LGtime),3])

    for jj in range(3):
        f = scipy.interpolate.interp1d(HGtime,dum[:,jj])
        resamp_HG[:,jj] = f(LGtime)

    # Cross-correlate the y-components. The search is bounded to align_shift
    # samples: the two accelerometers sit in one unit on one clock, so the true
    # offset is a sample or two, and searching every lag only offers chances to
    # land on a spurious match.
    align_shift = 1000
    [lags,corr_arr] = xcorr_fun(acc_lg[:,1],resamp_HG[:,1],align_shift)
    lag = int(lags[np.argmax(corr_arr)])

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

def estIMU_FC_MS_walking(acc, gyr, t,
              jerk_thresh=1e5,
              pre_swing_win=0.35,
              min_swing_depth=-3.0,
              post_gyro_int_win=0.40,
              post_gyro_int_thresh=50.0,
              az_peak_win=0.15,
              min_gap=1.05):
    """
    Detect foot contact (FC) events from a heel-mounted IMU, WALKING ONLY.

    The _walking suffix is load-bearing. min_gap defaults to a walking stride
    and every threshold below was tuned on treadmill walking at 0.6-1.6 m/s
    across -15 to +15 deg. It has NOT been validated for running, trail
    running, tennis, ski or snowboard, and the scripts for those keep their
    own detectors deliberately. Do not repoint them at this one without
    validating first.

    Originally tuned on treadmill walking across slopes (−15° to +15°) and
    speeds 0.6-1.6 m/s. Validated in the field against XSENSOR insoles on the
    Scarpa trek trail (Sept 2026): it resolves >94%% of steps uphill and
    downhill, and stride time is unbiased to within a few ms. See the Notes on
    what the detected event means on steep uphill terrain.

    Detection pipeline (4 steps)
    -----------------------------
    1. **Jerk peak** — squared jerk of the filtered ||acc|| norm
       (np.gradient(||acc||, t))² exceeds *jerk_thresh*. All three
       accelerometer axes contribute, making the detector robust to changes in
       the dominant impact axis with slope and contact style.

    2. **Swing confirmation** — the fore-aft acceleration (az) must dip
       below *min_swing_depth* within *pre_swing_win*
       seconds before the candidate. This confirms the foot was in forward
       swing immediately before contact.

    3. **Gyro integral confirmation** — the integral of gyroscope norm
       (total angular displacement, degrees) over *post_gyro_int_win* seconds
       after the candidate must be below *post_gyro_int_thresh*. A foot
       settling into stance quickly becomes rotationally quiet, integrating
       to a small total angle; a spurious mid-swing event keeps the foot
       rotating, accumulating far more angular displacement over the same
       window. This is more robust than a mean check because it captures
       the *duration* of post-event rotation, not just its instantaneous
       magnitude.

    4. **az local maximum re-anchor** — the candidate index is refined to the
       peak of az_corr within ±*az_peak_win* seconds. This centres every
       detection on the peak fore-aft deceleration at initial contact,
       giving a consistent biomechanical landmark regardless of slope or speed.

    Parameters
    ----------
    acc : numpy array (N, 3)
        Filtered X/Y/Z acceleration (m/s²). Apply a 2nd-order 50 Hz
        Butterworth low-pass filter before passing in (see :func:`filtIMUsig`).
    gyr : numpy array (N, 3)
        Filtered X/Y/Z gyroscope (deg/s).
    t : numpy array (N,)
        Time vector (seconds).
    jerk_thresh : float, optional
        Minimum squared jerk of ||acc|| norm to consider as a candidate event.
        Default 1e5. **Tuning knob** — increase to reduce false positives on
        noisy trials; decrease if contacts are being missed.
    pre_swing_win : float, optional
        Duration (s) of the window before the candidate searched for a
        negative az trough. Default 0.35 s.
    min_swing_depth : float, optional
        az_corr must dip below this value (m/s²) in pre_swing_win to pass
        swing confirmation. Default −3.0 m/s² (optimised across slopes
        −15° to +15° and speeds 0.6–1.6 m/s; original value was −1.0 m/s²).
        **Tuning knob** — make less negative to relax the check.
    post_gyro_int_win : float, optional
        Duration (s) of the post-contact window over which the gyroscope norm
        is integrated to verify stance. Default 0.40 s. A longer window better
        discriminates spurious mid-swing events (foot keeps rotating) from true
        contacts (foot quickly settles). **Tuning knob** — reduce only if the
        window overlaps the next stride at very fast cadences.
    post_gyro_int_thresh : float, optional
        Maximum integrated gyro norm (degrees = total angular displacement)
        in post_gyro_int_win to confirm stance. Default 50 deg. Derived from
        manual labelling of true vs spurious contacts across slopes −15° to
        +15° and speeds 0.6–1.6 m/s; good contacts accumulated < 35°,
        spurious contacts accumulated > 43°, giving a clean separation.
        **Tuning knob** — increase if contacts are missed at higher speeds.
    az_peak_win : float, optional
        Half-width (s) of the window around the confirmed candidate within
        which the az_corr local maximum is searched for re-anchoring.
        Default 0.15 s.
    min_gap : float, optional
        Minimum time (s) between successive detections; enforces a physiological
        lower bound on stride time. Default 1.05 s (optimised; original 0.75 s).
        **Tuning knob** — reduce only if studying conditions with stride times
        shorter than ~1.05 s (e.g. running or fast level walking).

    Returns
    -------
    FC : numpy array of int
        Sample indices of detected foot contact events.
    MS : numpy array of int
        Mid-stance sample indices, one per stride between successive FC.

    Notes
    -----
    * On steep uphill ground the detected event is the moment the foot comes to
      rest, which precedes the arrival of load by ~130-240 ms (measured against
      insoles, Sept 2026). The foot is genuinely stationary through that gap
      (gyro 32-45 deg/s, at or below mid-stance), so the event is the right
      landmark for a zero-velocity update, but it is NOT the instant of
      loading. Anything defined by load -- contact time, impulse, COM work
      phasing -- will read long uphill if taken from these events. Downhill the
      two coincide to within ~30 ms.
    * The function supersedes the original ``estIMU_HS_MS`` velocity-integral
      swing confirmation, which was unreliable on inclined surfaces.

    Examples
    --------
    >>> FC, MS = estIMU_FC_MS_walking(acc, gyr, t)
    """
    freq     = 1.0 / np.mean(np.diff(t))
    acc = filtIMUsig(acc, 50, t)
    gyr = filtIMUsig(gyr, 50, t)
    acc_norm = np.linalg.norm(acc, axis=1)
    gyr_norm = np.linalg.norm(gyr, axis=1)

    # Step 1 signal: squared jerk of acc norm (all axes)
    jerk_sig = (np.gradient(acc_norm, t)) ** 2

    # Fore-aft channel for steps 2 and 4, straight from the sensor. The belt
    # offset this used to remove does not exist overground.
    az_corr = acc[:, 2]

    pre_samp  = int(pre_swing_win     * freq)
    int_samp  = int(post_gyro_int_win * freq)
    peak_samp = int(az_peak_win       * freq)
    gap_samp  = int(min_gap           * freq)
    start     = int(0.5 * freq)
    end       = len(jerk_sig) - int_samp - 10

    FC = []
    jj = start

    while jj < end:
        if jerk_sig[jj] > jerk_thresh:

            # Step 1: refine to local jerk peak in short forward window
            wh  = min(end, jj + int(0.3 * freq))
            idx = int(np.argmax(jerk_sig[jj:wh])) + jj

            # Step 2: swing confirmation — negative az trough before candidate
            sw_lo    = max(0, idx - pre_samp)
            swing_ok = np.min(az_corr[sw_lo:idx]) < min_swing_depth

            # Step 3: gyro integral confirmation — total angular displacement
            # after candidate must be small (foot settled into stance)
            int_end   = min(len(gyr_norm), idx + int_samp)
            gyro_int  = np.trapezoid(gyr_norm[idx:int_end], t[idx:int_end])
            stance_ok = gyro_int < post_gyro_int_thresh

            if swing_ok and stance_ok:
                # Step 4: re-anchor to local az maximum (peak fore-aft decel)
                pk_lo  = max(0,                idx - peak_samp)
                pk_hi  = min(len(az_corr) - 1, idx + peak_samp)
                fc_idx = int(np.argmax(az_corr[pk_lo:pk_hi])) + pk_lo
                FC.append(fc_idx)
                jj = fc_idx + gap_samp
            else:
                jj = idx + 1

        jj += 1
    
    # Midstance detection
    gyr_energy = (np.linalg.norm(gyr, axis=1)) ** 2

    idx    = np.linalg.norm(acc, axis=1) > 2.5 * 9.81
    MS_sig = gyr_energy.copy()
    MS_sig[idx] = 1e6
    
    MS = np.array([
        (np.argmin(MS_sig[FC[jj] + 10:
                          FC[jj] + int((FC[jj + 1] - FC[jj]) * 0.2)])
         + FC[jj] + 10)
        for jj in range(len(FC) - 1)
    ])
    FC = np.array(FC[:-1])

    return (FC, MS)

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
        coordinate system (deg)

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

def intp_strides(var, landings, GS):
    """
    Function to interpolate the variable of interest across a stride
    (from foot contact to subsequent foot contact) in order to plot the
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    landings : list
        Foot contact indices
    GS : list or numpy array
        Indices of good strides to interpolate

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(GS)-1))
    # Index through the strides
    for ii in range(len(GS)-1):
        dum = var[landings[GS[ii]]:landings[GS[ii]+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))

    return intp_var

def computeRunSpeedIMU(acc, gyr, HS, MS, GS, t):
    """
    Estimate running speed from IMU signals.

    This function rotates the acceleration signal into an inertial coordinate
    system based on the integrand of the gyroscope signal. This rotation is
    performed in order to account for the gravitational acceleration. The
    corrected inertial coordinate system acceleration is then double
    integrated to estimate the displacement. The integrations are linearly
    corrected. Note that all time continuous signals span from foot contact
    to the subsequent midstance. The final displacement calculation is from
    foot contact to the subsequent foot contact.

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

    for count, jj in enumerate(GS):
        # Obtain the acceleration and gyroscope from foot contact (HS) to the
        # subsequent midstance (MS). MS is needed for linear drift correction
        acc_stride_plus = acc_filt[HS[jj]:MS[jj+1],:]
        gyr_stride_plus = gyr_filt[HS[jj]:MS[jj+1],:]
        time_stride_plus = t[HS[jj]:MS[jj+1]]
        time_stride = t[HS[jj]:HS[jj+1]]

        # Need the midstance and next foot contact indices
        MS_idx = MS[jj]-HS[jj]
        HS_idx = HS[jj+1]-HS[jj]

        # Compute the unit vector pointing in the approximate direction of gravity
        Fflat_accel = np.mean(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 0)/np.mean(np.linalg.norm(acc_stride_plus[MS_idx:MS_idx+MS_frames,:],axis = 1)).T

        # Rotation to the gravity vector: provides initial condition for gyro integration
        thetai = findRotToLab(Fflat_accel)

        # Integrate and linearly correct the gyroscope
        theta = cumtrapz(gyr_stride_plus,time_stride_plus,initial=0,axis=0)
        theta = theta-theta[MS_idx,:] # Midstance should be zero
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
           theta = theta-(thetai+thetai_up)/2 # average between the previous step
           thetai_up = thetai

        # Convert to radians
        theta = theta*np.pi/180
        # Rotate the accelerometer into the lab coordinate system (LCS)

        # Note: the rotation is built for every sample at once. It was previously a
        # list comprehension that assembled three 3x3 matrices per sample, which cost
        # ~19 s on an 18-min field file against ~1.4 s here. Same maths, same result
        # (verified bit-identical over 1639 strides on two trek trials, Sept 2026).
        cx, sx = cos(theta[:,0]), sin(theta[:,0])
        cy, sy = cos(theta[:,1]), sin(theta[:,1])
        cz, sz = cos(theta[:,2]), sin(theta[:,2])
        zz, oo = np.zeros_like(cx), np.ones_like(cx)
        Rz = np.stack([np.stack([cz,-sz,zz],-1),np.stack([sz,cz,zz],-1),np.stack([zz,zz,oo],-1)],1)
        Rx = np.stack([np.stack([oo,zz,zz],-1),np.stack([zz,cx,-sx],-1),np.stack([zz,sx,cx],-1)],1)
        Ry = np.stack([np.stack([cy,zz,sy],-1),np.stack([zz,oo,zz],-1),np.stack([-sy,zz,cy],-1)],1)
        acc_stride_plus_LCS = np.einsum('nij,nj->ni',Rz@Rx@Ry,acc_stride_plus)-np.array([9.81,0,0])

        # Integrate to get IMU velocity in the LCS
        IMU_vel = cumtrapz(acc_stride_plus_LCS,time_stride_plus,initial=0,axis=0)
        # Initial condition: IMU velocity is zero at midstance
        IMU_vel = IMU_vel-IMU_vel[MS_idx,:]
        # Remove drift
        slope = IMU_vel[-1,:]/(len(IMU_vel)-MS_idx-1)
        drift = (slope*np.ones([len(IMU_vel),3]))*(np.array([range(len(IMU_vel)),range(len(theta)),range(len(IMU_vel))])).T
        drift = drift-drift[MS_idx,:]
        IMU_vel = IMU_vel-drift
        # Integrate to get step length in the LCS
        IMU_SL = np.trapezoid(IMU_vel[0:HS_idx,:],time_stride,axis=0)
        run_speed.append(np.linalg.norm(IMU_SL)/(time_stride[-1]-time_stride[0]))

    return run_speed
