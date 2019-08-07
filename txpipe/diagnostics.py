from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile, TomographyCatalog
from .utils.stats import ParallelStatsCalculator, combine_variances
import numpy as np

class TXDiagnostics(PipelineStage):
    """
    """
    name='TXDiagnostics'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('g_psf_T', PNGFile),
        ('g_psf_g', PNGFile),
        ('g1_hist', PNGFile),
        ('g2_hist', PNGFile),
        ('g_snr', PNGFile),
        ('g_T', PNGFile),
        ('snr_hist', PNGFile),

    ]
    config = {}

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')

        # Collect together all the methods on this class called self.plot_*
        # They are all expected to be python coroutines - generators that
        # use the yield feature to pause and wait for more input.
        # We instantiate them all here
        plotters = [getattr(self, f)() for f in dir(self) if f.startswith('plot_')]

        # Start off each of the plotters.  This will make them all run up to the
        # first yield statement, then pause and wait for the first chunk of data
        for plotter in plotters:
            print(plotter)
            plotter.send(None)

        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = 10000
        shear_cols = ['mcal_psf_g1', 'mcal_psf_g2','mcal_g1','mcal_g2','mcal_psf_T_mean','mcal_s2n','mcal_T']
        iter_shear = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        tomo_cols = ['source_bin','lens_bin']
        iter_tomo = self.iterate_hdf('tomography_catalog','tomography',tomo_cols,chunk_rows)
        star_cols = ['measured_e1','model_e1','measured_e2','model_e2','measured_T','model_T']
        iter_star = self.iterate_hdf('star_catalog','stars',star_cols,chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data), (_, _, data2), in zip(iter_shear,iter_tomo):
            print(f"Read data {start} - {end}")
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.
            data.update(data2)
            #data.update(data3)
            for plotter in plotters:
                plotter.send(data)

        # Tell all the plotters to finish, collect together results from the different
        # processors, and make their final plots.  Plotters need to respond
        # to the None input and
        for plotter in plotters:
            try:
                plotter.send(None)
            except StopIteration:
                pass

    def plot_psf_shear(self):
        # mean shear in bins of PSF
        print("Making PSF shear plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        size = 11
        psf_g_edges = np.linspace(-5e-5, 5e-5, size+1)
        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])
        calc11 = ParallelStatsCalculator(size)
        calc12 = ParallelStatsCalculator(size)
        calc21 = ParallelStatsCalculator(size)
        calc22 = ParallelStatsCalculator(size)
        mu1 = ParallelStatsCalculator(size)
        mu2 = ParallelStatsCalculator(size)
        while True:
            data = yield

            if data is None:
                break
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1
            b1 = np.digitize(data['mcal_psf_g1'][qual_cut], psf_g_edges) - 1
            b2 = np.digitize(data['mcal_psf_g2'][qual_cut], psf_g_edges) - 1

            for i in range(size):
                w1 = np.where(b1==i)
                w2 = np.where(b2==i)

                # Do more things here to establish
                calc11.add_data(i, data['mcal_g1'][qual_cut][w1])
                calc12.add_data(i, data['mcal_g2'][qual_cut][w1])
                calc21.add_data(i, data['mcal_g1'][qual_cut][w2])
                calc22.add_data(i, data['mcal_g2'][qual_cut][w2])
                mu1.add_data(i, data['mcal_psf_g1'][qual_cut][w1])
                mu2.add_data(i, data['mcal_psf_g2'][qual_cut][w2])
        count11, mean11, var11 = calc11.collect(self.comm, mode='gather')
        count12, mean12, var12 = calc12.collect(self.comm, mode='gather')
        count21, mean21, var21 = calc21.collect(self.comm, mode='gather')
        count22, mean22, var22 = calc22.collect(self.comm, mode='gather')

        _, mu1, _ = mu1.collect(self.comm, mode='gather')
        _, mu2, _ = mu2.collect(self.comm, mode='gather')

        if self.rank != 0:
            return

        std11 = np.sqrt(var11/count11)
        std12 = np.sqrt(var12/count12)
        std21 = np.sqrt(var21/count21)
        std22 = np.sqrt(var22/count22)

        fig = self.open_output('g_psf_g', wrapper=True)
        dx = 0.1*(mu1[1] - mu1[0])

        slope11, intercept11, r_value11, p_value11, std_err11 = stats.linregress(mu1+dx,mean11)
        line11 = slope11*(mu1+dx)+intercept11

        slope12, intercept12, r_value12, p_value12, std_err12 = stats.linregress(mu1+dx,mean12)
        line12 = slope12*(mu1+dx)+intercept12

        slope21, intercept21, r_value21, p_value21, std_err21 = stats.linregress(mu2-dx,mean21)
        line21 = slope21*(mu2-dx)+intercept21

        slope22, intercept22, r_value22, p_value22, std_err22 = stats.linregress(mu2-dx,mean22)
        line22 = slope22*(mu2-dx)+intercept22

        plt.subplot(2,1,1)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (mean11 - flat1) / std11
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(mean11) - 1)

        plt.plot(mu1+dx,line11,color='blue',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
        plt.plot(mu1+dx,[0]*len(line11),color='black')

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (mean12 - flat1) / std12
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(mean12) - 1)

        plt.plot(mu1-dx,line12,color='red',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
        plt.plot(mu1-dx,[0]*len(line12),color='black')
        plt.errorbar(mu1+dx, mean11, std11, label='g1', fmt='+',color='blue')
        plt.errorbar(mu1-dx, mean12, std12, label='g2', fmt='+',color='red')
        plt.xlabel("PSF g1")
        plt.ylabel("Mean g")
        plt.legend()


        plt.subplot(2,1,2)
        
        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (mean21 - flat1) / std21
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(mean21) - 1)

        plt.plot(mu1+dx,line21,color='blue',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
        plt.plot(mu1+dx,[0]*len(line21),color='black')

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (mean22 - flat1) / std22
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(mean22) - 1)

        plt.plot(mu1-dx,line22,color='red',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
        plt.plot(mu1-dx,[0]*len(line22),color='black')
        plt.errorbar(mu1+dx, mean21, std21, label='g1', fmt='+',color='blue')
        plt.errorbar(mu1-dx, mean22, std22, label='g2', fmt='+',color='red')
        plt.xlabel("PSF g1")
        plt.ylabel("Mean g")
        plt.legend()

        # This also saves the figure
        fig.close()

    def plot_psf_size_shear(self):
        # mean shear in bins of PSF
        print("Making PSF size plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        size = 11
        psf_g_edges = np.linspace(0.2, 1.0, size+1)
        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
            
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1

            b1 = np.digitize(data['mcal_psf_T_mean'][qual_cut], psf_g_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][qual_cut][w])
                calc2.add_data(i, data['mcal_g2'][qual_cut][w])
                mu.add_data(i, data['mcal_psf_T_mean'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)

        dx = 0.05*(psf_g_mid[1] - psf_g_mid[0])
        if self.rank == 0:
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(mu+dx,mean1)
            line1 = slope1*(mu+dx)+intercept1
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(mu-dx,mean2)
            line2 = slope2*(mu-dx)+intercept2

            fig = self.open_output('g_psf_T', wrapper=True)

            # compute the mean and the chi^2/dof
            flat1 = 0
            z = (mean1 - flat1) / std1
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean1) - 1)
            plt.plot(mu+dx,line1,color='blue',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.plot(mu+dx,[0]*len(mu+dx),color='black')
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='blue')
            plt.legend(loc='best')

            # compute the mean and the chi^2/dof
            flat1 = 0
            z = (mean2 - flat1) / std2
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean2) - 1)
            plt.plot(mu-dx,line2,color='red',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='red')
            plt.xlabel("PSF T")
            plt.ylabel("Mean g")
            plt.legend(loc='best')
            plt.tight_layout()
            fig.close()

    def plot_snr_shear(self):
        # mean shear in bins of snr
        print("Making mean shear SNR plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        size = 10
        snr_edges = np.linspace(0,1000,size+1)
        snr_mid = 0.5*(snr_edges[1:] + snr_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1

            b1 = np.digitize(data['mcal_s2n'][qual_cut], snr_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][w])
                calc2.add_data(i, data['mcal_g2'][w])
                mu.add_data(i, data['mcal_s2n'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)

        dx = 0.05*(snr_mid[1] - snr_mid[0])
        if self.rank == 0:
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(mu+dx,mean1)
            line1 = slope1*(mu+dx)+intercept1
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(mu+dx,mean2)
            line2 = slope2*(mu+dx)+intercept2
            fig = self.open_output('g_snr', wrapper=True)
            # compute the mean and the chi^2/dof
            flat1 = 0
            z = (mean1 - flat1) / std1
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean1) - 1)
            plt.plot(mu+dx,line1,color='blue',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.plot(mu+dx,[0]*len(mu+dx),color='black')
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='blue')

            # compute the mean and the chi^2/dof
            flat1 = 0
            z = (mean2 - flat1) / std2
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean2) - 1)
            plt.plot(mu-dx,line2,color='red',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.plot(mu+dx,[0]*len(mu+dx),color='black')
            plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='red')
            plt.xlabel("SNR")
            plt.ylabel("Mean g")
            plt.legend()
            plt.tight_layout()
            fig.close()

    def plot_size_shear(self):
        # mean shear in bins of galaxy size
        print("Making mean shear galaxy size plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        size = 10
        T_edges = np.linspace(0,1,size+1)
        T_mid = 0.5*(T_edges[1:] + T_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1

            b1 = np.digitize(data['mcal_T'][qual_cut], T_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][qual_cut][w])
                calc2.add_data(i, data['mcal_g2'][qual_cut][w])
                mu.add_data(i, data['mcal_T'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)

        dx = 0.05*(T_mid[1] - T_mid[0])
        if self.rank == 0:
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(mu+dx,mean1)
            line1 = slope1*(mu+dx)+intercept1
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(mu+dx,mean2)
            line2 = slope2*(mu+dx)+intercept2
            fig = self.open_output('g_T', wrapper=True)

            flat1 = 0
            z = (mean1 - flat1) / std1
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean1) - 1)
            plt.plot(mu+dx,line1,color='blue',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.plot(mu+dx,[0]*len(mu+dx),color='black')
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='blue')

            flat1 = 0
            z = (mean2 - flat1) / std2
            chi2 = np.sum(z ** 2)
            chi2dof = chi2 / (len(mean2) - 1)
            plt.plot(mu-dx,line2,color='red',label='$\chi^2/dof = $'+str(np.round(chi2dof,5)))
            plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='red')
            plt.xlabel("galaxy size T")
            plt.ylabel("Mean g")
            plt.legend()
            plt.tight_layout()
            fig.close()


    def plot_g_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting histogram')
        import matplotlib.pyplot as plt
        from scipy import stats
        bins = 50
        edges = np.linspace(-1, 1, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        calc2 = ParallelStatsCalculator(bins)
        
        
        while True:
            data = yield

            if data is None:
                break
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1
        
            b1 = np.digitize(data['mcal_g1'][qual_cut], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][qual_cut][w])
                calc2.add_data(i, data['mcal_g2'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)
        if self.rank != 0:
            return
        fig = self.open_output('g1_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.xlabel("g1")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

        fig = self.open_output('g2_hist', wrapper=True)
        plt.bar(mids, count2, width=edges[1]-edges[0], align='center',edgecolor='black',color='purple')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("g2")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count2))
        fig.close()

    def plot_snr_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting snr histogram')
        import matplotlib.pyplot as plt
        bins = 50
        edges = np.linspace(0, 10, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
            qual_cut &= data['lens_bin'] !=-1

            b1 = np.digitize(np.log10(data['mcal_s2n'][qual_cut]), edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, np.log10(data['mcal_s2n'][qual_cut][w]))

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        if self.rank != 0:
            return
        fig = self.open_output('snr_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("log(snr)")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()