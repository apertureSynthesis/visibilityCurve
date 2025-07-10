import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib

class visibilityCurve(object):

    def __init__(self,obsVis,parentVis,daughterVis,binWid,binOffset,lpString,titleString,tp=None,tpe=None,uvmax=None):
        self.obsVis = obsVis
        self.parentVis = parentVis
        self.daughterVis = daughterVis
        self.binWid = binWid
        self.binOffset = binOffset
        self.lpString = lpString
        self.titleString = titleString
        self.tp = tp
        self.tpe = tpe
        self.uvmax = uvmax
        self.c = 2.99792458e8

        

    def import_vis(self,file):
        vis = import_data_ms(file)
        self.f0 = 0.5 * (vis.freqs[-1] + vis.freqs[0])[0]
        self.dv = self.c * np.abs((vis.freqs[1] - vis.freqs[0])/self.f0) / 1000.
        re = np.squeeze(np.real(vis.VV))
        visRMS = np.nanstd(re)
        rms = np.zeros(len(re))+visRMS
        uv = np.sqrt(vis.uu**2 + vis.vv**2) * self.c / self.f0

        if self.uvmax != None:
            uvInds = np.where(uv <= self.uvmax)
            uv = uv[uvInds]
            re = re[uvInds]
            rms = rms[uvInds]

        return uv, re, rms

    def binVis(self, uv, re, rms, chans):
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
                    yerrs.append(1.29/np.sum((len(chans))/rms[binInds==i]**2)**0.5)     

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
        datachans = presum > 0.1

        #Bin them and then integrate from Jy to Jy km/s
        uvBinned, reBinned, yerrs = self.binVis(uv, re, rms, datachans)
        puvBinned, preBinned, pyerrs = self.binVis(puv, pre, prms, datachans)
        duvBinned, dreBinned, dyerrs = self.binVis(duv, dre, drms, datachans)

        #Generate interpolated curves for the model visibilities
        pCurve = interp1d(puvBinned,preBinned,kind='cubic',fill_value='extrapolate',bounds_error=False)
        dCurve = interp1d(duvBinned,dreBinned,kind='cubic',fill_value='extrapolate',bounds_error=False)

        uvInterp = np.arange(np.min(uvBinned),np.max(uvBinned),(np.max(uvBinned)-np.min(uvBinned))/100.)
        uvsInterp = [i / (1e3*self.c/(self.f0)) for i in uvInterp]

        pInterp = pCurve(uvInterp)
        dInterp = dCurve(uvInterp)

        #Generate baselines in kilo-lambda for plotting
        uvsBinned = [i / (1e3*self.c/(self.f0)) for i in uvBinned]

        reBinned *= self.dv * len(datachans)
        yerrs *= self.dv * len(datachans)
        preBinned *= self.dv * len(datachans)
        dreBinned *= self.dv * len(datachans)
        pInterp *= self.dv * len(datachans)
        dInterp *= self.dv * len(datachans)

        fig, ax = plt.subplots(figsize=(10,10))

        #Set some plotting options
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['mathtext.default'] = 'regular'
        matplotlib.rcParams['font.weight'] = 'bold'
        matplotlib.rcParams['axes.labelweight'] = 'bold'

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

        #ax.legend()
        h,l = ax.get_legend_handles_labels()
        h = [h[2],h[0],h[1]]
        l = [l[2],l[0],l[1]]
        ax.legend(h,l,fontsize=20)

        plt.tight_layout()
        plt.savefig('K2.jpg')
        plt.show()