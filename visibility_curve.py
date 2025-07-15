import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from visibilityCurve.vis_sample import import_data_ms

class visibilityCurve(object):

    """
    A class that plots a visibility curve from an observed ALMA comet Measurement Set
    along with parent and daughter models, also stored as Measurement Sets.

    Inputs:
    obsVis: observed ALMA Measurement Set
    parentVis: parent (Haser) model Measurement Set
    daughterVis: daughter (Haser) model Measurement Set
    binWid: binning width (in meters) for uv-distance
    binoffset: offset the binning from 0 if desired (m)
    lpString: string describing the label for the daughter model (presumably Lp)
    titleString: string for the title of the plot
    figName: string for the name of the output plot file. must include desired file extension
    tp: optionally add total power data and model (TBD)
    tpe: optionally add errors on total power data and model (TBD)
    uvmax: optionally cut off visibilities at a maximum uv-distance (m)

    Outputs:
    a plot showing the observed and modeled binned visibilities with labels
    """
    def __init__(self,obsVis,parentVis,daughterVis,binWid,binOffset,lpString,titleString,figName,chan1=None,chan2=None,tp=None,tpe=None,uvmax=None):
        self.obsVis = obsVis
        self.parentVis = parentVis
        self.daughterVis = daughterVis
        self.binWid = binWid
        self.binOffset = binOffset
        self.lpString = lpString
        self.titleString = titleString
        self.figName = figName
        self.chan1 = chan1
        self.chan2 = chan2
        self.tp = tp
        self.tpe = tpe
        self.uvmax = uvmax
        self.c = 2.99792458e8

        

    def import_vis(self,file):
        #Read in a measurement set using Ryan Loomis' vis_sample
        vis = import_data_ms(file)
        self.f0 = 0.5 * (vis.freqs[-1] + vis.freqs[0])[0]
        self.dv = self.c * np.abs((vis.freqs[1] - vis.freqs[0])/self.f0) / 1000.
        print(self.dv,self.f0/1e9)
        #Real part of the visibilities
        re = np.squeeze(np.real(vis.VV))
        #Measured noise on visibilities
        visRMS = np.nanstd(re)
        rms = np.zeros(len(re))+visRMS
        #uv-distances
        uv = np.sqrt(vis.uu**2 + vis.vv**2) * self.c / self.f0

        #Cut off at a maximum uv-distance if desired
        if self.uvmax != None:
            uvInds = np.where(uv <= self.uvmax)
            uv = uv[uvInds]
            re = re[uvInds]
            rms = rms[uvInds]

        return uv, re, rms

    def binVis(self, uv, re, rms, chans):
        #Bin the visibilities
        bins = np.arange(self.binOffset, uv.max(), self.binWid)
        binInds = np.digitize(uv,bins)

        uvBinned = []
        yerrs = []

        for i in range(len(bins)):
            if (np.sum(binInds==i) > 0):
                uvBinned.append(np.average(uv[binInds==i]))
                if len(re.shape) < 2:
                    yerrs.append(1.29/np.sum(1/rms[binInds==i]**2)**0.5)
                else:
                    yerrs.append(1.29/np.sum((len(np.where(chans==True)[0]))/rms[binInds==i]**2)**0.5)     

        reBinned = []
        for i in range(len(bins)):
            if (np.sum(binInds==i) > 0):
                reBinned.append(np.average(re[:,chans][binInds==i]))

        return np.array(uvBinned), np.array(reBinned), np.array(yerrs)

    def plotCurve(self):

        #Import the observed visibilities and modeled visibilities and find the line-containing channels
        uv, re, rms = self.import_vis(self.obsVis)
        puv, pre, prms = self.import_vis(self.parentVis)
        duv, dre, drms = self.import_vis(self.daughterVis)

        # Get channels that contain data
        presum = np.sum(pre,axis=0)
        if self.chan1 == None:
            datachans = presum > 0.1
        else:
            datachans = np.full(len(presum), False)
            datachans[self.chan1:self.chan2+1] = True

        #Bin them and then integrate from Jy to Jy km/s
        uvBinned, reBinned, yerrs = self.binVis(uv, re, rms, datachans)
        puvBinned, preBinned, pyerrs = self.binVis(puv, pre, prms, datachans)
        duvBinned, dreBinned, dyerrs = self.binVis(duv, dre, drms, datachans)

        #Generate interpolated curves for the model visibilities
        pCurve = interp1d(puvBinned,preBinned,kind='cubic',fill_value='extrapolate',bounds_error=False)
        dCurve = interp1d(duvBinned,dreBinned,kind='cubic',fill_value='extrapolate',bounds_error=False)

        #Finer grid of baselines for plotting the interpolated models
        uvInterp = np.arange(np.min(uvBinned),np.max(uvBinned),(np.max(uvBinned)-np.min(uvBinned))/100.)

        pInterp = pCurve(uvInterp)
        dInterp = dCurve(uvInterp)

        #Generate baselines in kilo-lambda for plotting
        uvsBinned = [i / (1e3*self.c/(self.f0)) for i in uvBinned]
        uvsInterp = [i / (1e3*self.c/(self.f0)) for i in uvInterp]

        #Integrate everything
        nchan = len(np.where(datachans==True)[0])
        print(nchan)
        reBinned *= self.dv * nchan
        yerrs *= self.dv * nchan
        preBinned *= self.dv * nchan
        dreBinned *= self.dv * nchan
        pInterp *= self.dv * nchan
        dInterp *= self.dv * nchan

        #Make the figure
        fig, ax = plt.subplots(figsize=(10,10))

        ax.errorbar(uvsBinned, reBinned, yerr=yerrs, linestyle=' ',marker='o',markersize=8,elinewidth=1.5,ecolor='k',
                    capsize=5,label='Observed')
        ax.plot(uvsInterp,dInterp,linestyle='-',color='C1',label='Daughter Model: '+self.lpString)
        ax.plot(uvsInterp,pInterp,linestyle='--',color='C2',label='Parent Model')

        ax.tick_params(axis='x',direction='in',color='black',length=7,labelsize=21)
        ax.tick_params(axis='y',direction='in',color='black',length=7,labelsize=21)
        ax.tick_params(bottom=True,top=True,left=True,right=True)
        ax.tick_params(labelleft=True,labelbottom=True,labeltop=False,labelright=False)
        ax.set_xlabel('$uv$-distance (k$\lambda$)',fontsize=25,color='black')
        ax.set_ylabel('Re(Visibility) (Jy km s$^{-1}$)',fontsize=25)
        ax.set_title(self.titleString,fontsize=25,fontweight='bold')

        #Order the legend items
        h,l = ax.get_legend_handles_labels()
        h = [h[2],h[0],h[1]]
        l = [l[2],l[0],l[1]]
        ax.legend(h,l,fontsize=20)

        plt.tight_layout()
        plt.savefig(self.figName)
        plt.show()