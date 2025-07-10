import numpy as np

##import_data_ms from Ryan Loomis' vis_sample
##included manually to ease install problems
##https://github.com/AstroChem/vis_sample/blob/master/LICENSE
class Visibility:
    def __init__(self, VV, uu, vv, wgts, freqs, rfreq=None):
        c_kms = 2.99792458e5 # [km s^-1]
        self.VV = VV # [Jy]
        self.uu = uu # [lambda]
        self.vv = vv # [lambda]
        self.wgts = wgts # [Jy^-2]
        self.freqs = np.array(freqs) # [Hz]
        if rfreq:
            self.rfreq = rfreq # [Hz]
        else:
            self.rfreq = np.mean(self.freqs) # [Hz]
        self.vels = (self.rfreq - self.freqs)/self.rfreq*c_kms # [km/s]

# CASA interfacing code comes from Peter Williams' casa-python and casa-data package
# commands for retrieving ms data are from Sean Andrews
def import_data_ms(filename):
    """Imports data from a casa measurement set (ms) and returns Visibility object"""
    try:
        import casatools
    except ImportError:
        print("casatools was not able to be imported, make sure all dependent packages are installed")
        print("try instructions at https://casa.nrao.edu/casadocs/casa-5.6.0/introduction/casa6-installation-and-usage")
        sys.exit(1)

    tb = casatools.table()
    ms = casatools.ms()

    cc = 2.99792458e10 # [cm s^-1]
    
    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    tb.open(filename)
    data    = tb.getcol("DATA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    flags    = tb.getcol("FLAG")
    tb.close()
    

    # Use CASA ms tools to get the channel/spw info
    ms.open(filename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info["0"]["NumChan"]
    npol = spw_info["0"]["NumCorr"]
    ms.close()

    # Use CASA table tools to get frequencies
    tb.open(filename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    tb.close()

    tb.open(filename+"/SOURCE")
    rfreq = tb.getcol("REST_FREQUENCY")[0][0]
    tb.close()


    # break out the u, v spatial frequencies, convert from m to lambda
    uu = uvw[0,:]*rfreq/(cc/100)
    vv = uvw[1,:]*rfreq/(cc/100)

    # check to see whether the polarizations are already averaged
    data = np.squeeze(data)
    weight = np.squeeze(weight)
    flags = np.squeeze(flags)


    if npol==1:
        Re = data.real
        Im = data.imag
        wgts = weight

    else:
        # polarization averaging
        Re_xx = data[0,:].real
        Re_yy = data[1,:].real
        Im_xx = data[0,:].imag
        Im_yy = data[1,:].imag
        weight_xx = weight[0,:]
        weight_yy = weight[1,:]
        flags = flags[0,:]*flags[1,:]

        # - weighted averages
        with np.errstate(divide='ignore', invalid='ignore'):
            Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
            Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        wgts = (weight_xx + weight_yy)

    # toss out the autocorrelation placeholders
    xc = np.where(ant1 != ant2)[0]

    # check if there's only a single channel
    if nchan==1:
        data_real = Re[np.newaxis, xc]
        data_imag = Im[np.newaxis, xc]
        flags = flags[xc]
    else:
        data_real = Re[:,xc]
        data_imag = Im[:,xc]
        flags = flags[:,xc]

        # if the majority of points in any channel are flagged, it probably means someone flagged an entire channel - spit warning
        if np.mean(flags.all(axis=0)) > 0.5:
            print("WARNING: Over half of the (u,v) points in at least one channel are marked as flagged. If you didn't expect this, it is likely due to having an entire channel flagged in the ms. Please double check this and be careful if model fitting or using diff mode.")

        # collapse flags to single channel, because weights are not currently channelized
        flags = flags.any(axis=0)

    data_wgts = wgts[xc]
    data_uu = uu[xc]
    data_vv = vv[xc]

    data_VV = data_real+data_imag*1.0j

    #warning that flagged data was imported
    if np.any(flags):
        print("WARNING: Flagged data was imported. Visibility interpolation can proceed normally, but be careful with chi^2 calculations.")

    # now remove all flagged data (we assume the user doesn't want to interpolate for these points)
    # commenting this out for now, but leaving infrastructure in place if desired later
    #data_wgts = data_wgts[np.logical_not(flags)]
    #data_uu = data_uu[np.logical_not(flags)]
    #data_vv = data_vv[np.logical_not(flags)]
    #data_VV = data_VV[:,np.logical_not(flags)]

    return Visibility(data_VV.T, data_uu, data_vv, data_wgts, freqs, rfreq)

