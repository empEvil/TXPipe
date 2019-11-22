from .base_stage import PipelineStage
from .data_types import HDFFile, MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile, PNGFile
import numpy as np
import random
import collections
import sys

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'theta', 'value', 'cov', 'npair', 'weight', 'i', 'j'])

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2



class TXTwoPoint_Jacked(PipelineStage):
    name='TXTwoPoint_Jacked'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('KMcenters', KMcenters)
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'source_bins':[-1],
        'lens_bins':[-1],
        'reduce_randoms_size':1.0,
        'do_shear_shear': True,
        'do_shear_pos': True,
        'do_pos_pos': True,
        'ncen': 50 #Number of centers
        }

    def run(self):
        """
        Run the analysis for this stage.
        """
        import sacc
        import healpy
        import treecorr
        # Load the different pieces of data we need into
        # one large dictionary which we accumulate
        data = {}
        self.load_tomography(data)
        self.load_shear_catalog(data)
        self.load_random_catalog(data)
        # This one is optional - this class does nothing with it
        self.load_lens_catalog(data)
        # Binning information
        self.read_nbin(data)
        self.load_KM_centers(data)

        # Calculate metadata like the area and related
        # quantities
        meta = self.calculate_metadata(data)

        # Choose which pairs of bins to calculate
        calcs = self.select_calculations(data)

        sys.stdout.flush()

        # This splits the calculations among the parallel bins
        # It's not necessarily the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment,
        # but for this case I would expect it to be mostly fine
        results = []
        for i,j,k in self.split_tasks_by_rank(calcs):
            results += self.call_treecorr(data, i, j, k)

        # If we are running in parallel this collects the results together
        results = self.collect_results(results)

        # Save the results
        if self.rank==0:
            self.write_output(data, meta, results)


    def select_calculations(self, data):
        source_list = data['source_list']
        lens_list = data['lens_list']
        calcs = []

        # For shear-shear we omit pairs with j>i
        if self.config['do_shear_shear']:
            k = SHEAR_SHEAR
            for i in source_list:
                for j in range(i+1):
                    if j in source_list:
                        calcs.append((i,j,k))

        # For shear-position we use all pairs
        if self.config['do_shear_pos']:
            k = SHEAR_POS
            for i in source_list:
                for j in lens_list:
                    calcs.append((i,j,k))

        # For position-position we omit pairs with j>i
        if self.config['do_pos_pos']:
            if not 'random_bin' in data:
                raise ValueError("You need to have a random catalog to calculate position-position correlations")
            k = POS_POS
            for i in lens_list:
                for j in range(i+1):
                    if j in lens_list:
                        calcs.append((i,j,k))

        if self.rank==0:
            print(f"Running these calculations: {calcs}")

        return calcs

    def collect_results(self, results):
        if self.comm is None:
            return results

        results = self.comm.gather(results, root=0)

        # Concatenate results on master
        if self.rank==0:
            results = sum(results, [])

        return results

    def read_nbin(self, data):
        """
        Determine the bins to use in this analysis, either from the input file
        or from the configuration.
        """
        if self.config['source_bins'] == [-1] and self.config['lens_bins'] == [-1]:
            source_list, lens_list = self._read_nbin_from_tomography()
        else:
            source_list, lens_list = self._read_nbin_from_config()

        ns = len(source_list)
        nl = len(lens_list)
        print(f'Running with {ns} source bins and {nl} lens bins')

        data['source_list']  =  source_list
        data['lens_list']  =  lens_list


    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        tomo = self.open_input('tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_source = d['nbin_source']
        nbin_lens = d['nbin_lens']
        source_list = range(nbin_source)
        lens_list = range(nbin_lens)
        return source_list, lens_list

    def _read_nbin_from_config(self):
        # TODO handle the case where the user only specefies
        # bins for only sources or only lenses
        source_list = self.config['source_bins']
        lens_list = self.config['lens_bins']

        # catch bad input
        tomo_source_list, tomo_lens_list = self._read_nbin_from_tomography()
        tomo_nbin_source = len(tomo_source_list)
        tomo_nbin_lens = len(tomo_lens_list)

        nbin_source = len(source_list)
        nbin_lens = len(lens_list)

        if source_list == [-1]:
            source_list = tomo_source_list
        if lens_list == [-1]:
            lens_list = tomo_lens_list

        # if more bins are input than exist, raise an error
        if not nbin_source <= tomo_nbin_source:
            raise ValueError(f'Requested too many source bins in the config ({nbin_source}): max is {tomo_nbin_source}')
        if not nbin_lens <= tomo_nbin_lens:
            raise ValueError(f'Requested too many lens bins in the config ({nbin_lens}): max is {tomo_nbin_lens}')

        # make sure the bin numbers actually exist
        for i in source_list:
            if i not in tomo_source_list:
                raise ValueError(f"Requested source bin {i} that is not in the input file")

        for i in lens_list:
            if i not in tomo_lens_list:
                raise ValueError(f"Requested lens bin {i} that is not in the input file")

        return source_list, lens_list



    def write_output(self, data, meta, results):
        import sacc
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t

        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('photoz_stack')

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        for i in data['source_list']:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)

        # For both source and lens
        for i in data['lens_list']:
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            S.add_tracer('NZ', f'lens_{i}', z, Nz)
        # Closing n(z) file
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        for d in results:
            tracer1 = f'source_{d.i}' if d.corr_type in [XIP, XIM, GAMMAT] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [XIP, XIM] else f'lens_{d.j}'


            # Each of our Measurement objects contains various theta values,
            # and we loop through and add them all
            n = len(d.value)
            for i in range(n):
                error = np.sqrt(d.cov[i,i])
                S.add_data_point(d.corr_type, (tracer1,tracer2), d.value[i],
                    theta=d.theta[i], error=error, npair=d.npair[i], weight=d.weight[i])

        #self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data'), overwrite=True)

    def write_metadata(self, S, meta):
        # We also save the associated metadata to the file
        for k,v in meta.items():
            if np.isscalar(v):
                S.metadata[k] = v
            else:
                for i, vi in enumerate(v):
                    S.metadata[f'{k}_{i}'] = vi

        # Add provenance metadata.  In managed formats this is done
        # automatically, but because the Sacc library is external
        # we do it manually here.
        for key, value in self.gather_provenance().items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    S.metadata[f'provenance/{key}_{i}'] = v
            else:
                S.metadata[f'provenance/{key}'] = value


    def call_treecorr(self, data, i, j, k):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        WTHETA = sacc.standard_types.galaxy_density_xi

        results = []

        if k==SHEAR_SHEAR:
            theta, xip, xim, xip_cov, xim_cov, npairs, weight = self.calculate_shear_shear(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(XIP, theta, xip, xip_cov, npairs, weight, i, j))
            results.append(Measurement(XIM, theta, xim, xim_cov, npairs, weight, i, j))

        elif k==SHEAR_POS:
            theta, val, cov, npairs, weight = self.calculate_shear_pos(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(GAMMAT, theta, val, cov, npairs, weight, i, j))

        elif k==POS_POS:
            theta, val, cov, npairs, weight = self.calculate_pos_pos(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(WTHETA, theta, val, cov, npairs, weight, i, j))
        sys.stdout.flush()
        return results


    def get_m(self, data, i):
        """
        Calculate the metacal correction factor for this tomographic bin.
        # TODO: Add the selection bias R_S?  Or check if it is added elsewhere.
        """

        mask = (data['source_bin'] == i)

        m1 = np.mean(data['r_gamma'][mask][:,0,0]) # R11, taking the mean for the bin, TODO check if that's what we want to do
        m2 = np.mean(data['r_gamma'][mask][:,1,1]) #R22

        return m1, m2, mask


    def get_shear_catalog(self, data, i):
        import treecorr
        m1,m2,mask = self.get_m(data, i)

        g1 = data['mcal_g1'][mask]
        g2 = data['mcal_g2'][mask]
#I think this is where we need the splitting up, such that cat becomes a n x n array instead of just one catalog.
        cat = treecorr.Catalog(
            g1 = (g1 - g1.mean()) / m1,
            g2 = (g2 - g2.mean()) / m2,
            ra = data['ra'][mask],
            dec = data['dec'][mask],
            ra_units='degree', dec_units='degree')
        ncen = self.config['ncen']
        cat2 = np.empty(ncen, dtype = treecorr.Catalog)
        km = KMeans(data['KMcenters'])
        labels = data['jack_label'][mask]
        for j in range(ncen):
            cat2[j] = treecorr.Catalog(
            g1 = cat.g1[labels== j],
            g2 = cat.g2[labels== j],
            ra = cat.ra[labels== j],
            dec = cat.dec[labels == j],
            ra_units = 'degree', dec_units ='degree')
        return cat2


    def get_lens_catalog(self, data, i):
        import treecorr


        mask = data['lens_bin'] == i

        if 'lens_ra' in data:
            ra = data['lens_ra'][mask]
            dec = data['lens_dec'][mask]
        else:
            ra = data['ra'][mask]
            dec = data['dec'][mask]
#Again a spot for the split.
        cat = treecorr.Catalog(
            ra=ra, dec=dec,
            ra_units='degree', dec_units='degree')

        ncen = self.config['ncen']
        cat2 = np.empty(ncen, dtype = treecorr.Catalog)
        km = KMeans(data['KMcenters'])
        labels = data['jack_label'][mask]#km.find_nearest((cat.ra, cat.dec).T, ncen, maxiter = 100)
        for j in range(ncen):
            cat2[j] = treecorr.Catalog(
            ra = cat.ra[labels== j],
            dec = cat.dec[labels == j],
            ra_units = 'degree', dec_units ='degree')

        if 'random_bin' in data:
            random_mask = data['random_bin']==i
            rancat  = treecorr.Catalog(
                ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                ra_units='degree', dec_units='degree')
            rancat2 = np.empty(ncen, dtype = treecorr.Catalog)
            for j in range(ncen):
                rancat2[j] = treecorr.Catalog(ra = rancat.ra[labels == j],
                dec = rancat.dec[labels == j],
                ra_units = 'degree', dec_units = 'degree')
        else:
            rancat2 = None

        return cat2, rancat2


    def calculate_shear_shear(self, data, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(data, i)
        #This is removed for now, cause I don't know about it for the split up cats
        n_i = 0#cat_i.nobj


        if i==j:
            cat_j = cat_i
            n_j = n_i
        else:
            cat_j = self.get_shear_catalog(data, j)
            #This is removed for now, cause I don't know about it for the split up cats
            n_j = 0#cat_j.nobj


        print(f"Rank {self.rank} calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects")
# THis is the new thing added:
        ncen = self.config['ncen']
        nbins = self.config['nbins']
        gg = N.full([ncen,ncen], treecorr.GGCorrelation(self.config), dtype = treecorr.GGCorrelation)
        gg2 = N.full(ncen, treecorr.GGCorrelation(self.config), dtype = treecorr.GGCorrelation)
        index = np.zeros(ncen)
        for m in range(ncen):
            index[m] = m
            for n in range(ncen):
                gg[m,n].process_cross(cat_i[m],cat_j[n])


        xip = np.zeros((nbins,ncen))
        xim = np.zeros_like(xip)
        xipvar = np.zeros_like(xip)
        ximvar = np.zeros_lip(xim)
        meanlogr = np.zeros_like(xip)

        gg_pairs = np.zeros(ncen)
        gg_weight = np.zeros(ncen)
        for m in range(ncen):
            #gg2[m] = treecorr.GGCorrelation(self.config)

            for n in range(ncen-1):
                l = (m+n+1) % ncen
                for o in range(ncen-1):
                    p = (m+o+1) % ncen
                    gg2[m].__iadd__(gg[l,p])
            varG1 = treecorr.calculateVarG(cat_i[index != m])
            varG2 = treecorr.calculateVarG(cat_j[index != m])
            gg2[m].finalize(varG1,varG2)

            meanlogr[:,m] = gg2[m].meanlogr
            xip[:,m] = gg2[m].xip
            xim[:,m] = gg2[m].xim
            xipvar[:,m] = gg2[m].varxip
            ximvar[:,m] = gg2[m].varxim
            gg_pairs[m] = gg2[m].npairs
            gg_weight[m] = gg2[m].weight

        #Cool now we have ncen data so now we should probably try and calculate errors and stuff.
        xip2 = N.average(xip, axis = 1)
        xim2 = N.average(xim, axis = 1)
        xipvar2 = N.average(xipvar, axis = 1) #This is only the average of the shape variance
        ximvar2 = N.average(ximvar, axis = 1) #calculated in treecorr
        meanlogr2 = N.average(meanlogr, axis = 1)

        cov_xip = N.zeros((nbins,nbins))
        cov_xim = N.zeros((nbins,nbins))
        for l in range(nbins):
            for m in range(nbins):
                for n in range(ncen):
                    cov_xip[l,m] += (xip[l,n] - xip2[l])*(xip[m,n] - xip2[m])
                    cov_xim[l,m] += (xim[l,n] - xim2[l])*(xim[m,n] - xim2[m])
        cov_xip = (ncen-1)/ncen * cov_xip
        cov_xim = (ncen-1)/ncen * cov_xim

        theta = np.exp(meanlogr2)

        return theta, xip2, xim2, cov_xip, cov_xim, gg_npairs, gg_weight

    def calculate_shear_pos(self,data, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(data, i)
        #see the shear shear for reasoning
        n_i = 0 #cat_i.nobj

        cat_j, rancat_j = self.get_lens_catalog(data, j)
        n_j = 0 #cat_j.nobj
        n_rand_j = 0#rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")
# THIS WILL NEED TO CHANGE
        ncen = self.config['ncen']
        nbins = self.config['nbins']
        ng = N.full([ncen,ncen], treecorr.NGCorrelation(self.config), dtype = treecorr.NGCorrelation)
        ng2 = N.full(ncen, treecorr.NGCorrelation(self.config), dtype = treecorr.NGCorrelation)
        rg = N.full_like(ng)
        rg2 = N.full_like(ng2)
        index = np.zeros(ncen)
        for m in range(ncen):
            index[m] = m
            for n in range(ncen):
                ng[m,n].process_cross(cat_i[m],cat_j[n])
                if rancat_j:
                    rg[m,n].process_cross(rancat_j[m], cat_i[n])
                else:
                    rg[m,n] = None

        gammat = np.zeros((nbins,ncen))
        gammat-im = np.zeros_like(gammat)
        gammat_varr = np.zeros_like(gammat)
        ximvar = np.zeros_lip(gammat)
        meanlogr = np.zeros_like(gammat)


        ng_pairs = np.zeros(ncen)
        ng_weight = np.zeros(ncen)
        for m in range(ncen):
            #ng2[m] = treecorr.NGCorrelation(self.config)

            for n in range(ncen-1):
                l = (m+n+1) % ncen
                for o in range(ncen-1):
                    p = (m+o+1) % ncen
                    ng2[m].__iadd__(ng[l,p])
                    rg2[m].__iadd__(rg[l,p])
            varG1 = treecorr.calculateVarG(cat_i[index != m])
            ng2[m].finalize(varG1)
            rg2[m].finalize(varG1)

            meanlogr[:,m] = ng2[m].meanlogr
            gammat[:,m], gammat_im[:,m],  gammat_varr[:,m]= ng2[m].calculateXi(rg=rg2[m])
            ng_pairs[m] = ng2[m].npairs
            ng_weight[m] = ng2[m].weight
# ALSO CHANGE
        gammat2 = N.average(gammat, axis = 1)
        gammat_im2 = N.average(gammat_im, axis = 1)
        gammat_varr2 = N.average(gammat_varr, axis = 1) #This is only the average of the shape variance
        meanlogr2 = N.average(meanlogr, axis = 1)
        theta = np.exp(meanlogr2)

        cov_xi = N.zeros((nbins,nbins))
        cov_xi_im = N.zeros((nbins,nbins))
        for l in range(nbins):
            for m in range(nbins):
                for n in range(ncen):
                    cov_xi[l,m] += (gammat[l,n] - gammat2[l])*(gammat[m,n] - gammat2[m])
                    cov_xi_im[l,m] += (gammat_im[l,n] - gammat_im2[l])*(gammat_im[m,n] - gammat_im2[m])
        cov_xi = (ncen-1)/ncen * cov_xi
        cov_xi_im = (ncen-1)/ncen * cov_xi_im



        return theta, gammat2, cov_xi, ng_npairs, ng_weight


    def calculate_pos_pos(self, data, i, j):
        import treecorr

        cat_i, rancat_i = self.get_lens_catalog(data, i)
        #see shear shear for why its just zero
        n_i = 0#cat_i.nobj
        n_rand_i =0# rancat_i.nobj if rancat_i is not None else 0

        if i==j:
            cat_j = cat_i
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            cat_j, rancat_j = self.get_lens_catalog(data, j)
            n_j = 0#cat_j.nobj
            n_rand_j =0# rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating position-position bin pair ({i},{j}): {n_i} x {n_j} objects, "
            f"{n_rand_i} x {n_rand_j} randoms")

#ALL THESE SHOULD CHANGE!
        ncen = self.config['ncen']
        nbins = self.config['nbins']
        nn = N.full([ncen,ncen], treecorr.NGCorrelation(self.config), dtype = treecorr.NGCorrelation)
        nn2 = N.full(ncen, treecorr.NGCorrelation(self.config), dtype = treecorr.NGCorrelation)
        rn = N.full_like(nn)
        rn2 = N.full_like(nn2)
        nr = N.full_like(nn)
        nr2 = N.full_like(nn2)
        rr = N.full_like(nn)
        rr2 = N.full_like(nn2)

        index = np.zeros(ncen)
        for m in range(ncen):
            index[m] = m
            for n in range(ncen):
                nn[m,n].process_cross(cat_i[m],cat_j[n])
                nr[m,n].process_cross(cat_i[m],rancat_j[n])
                rn[m,n].process_cross(rancat_i[m], cat_j[n])
                rr[m,n].process_cross(rancat_i[m], rancat_j[n])



        wtheta = np.zeros((nbins,ncen))
        wthetaerr = np.zeros_like(wheta)


        nn_pairs = np.zeros(ncen)
        nn_weight = np.zeros(ncen)
        for m in range(ncen):
            #ng2[m] = treecorr.NNCorrelation(self.config)

            for n in range(ncen-1):
                l = (m+n+1) % ncen
                for o in range(ncen-1):
                    p = (m+o+1) % ncen
                    nn2[m].__iadd__(nn[l,p])
                    nr2[m].__iadd__(nr[l,p])
                    rn2[m].__iadd__(rn[l,p])
                    rr2[m].__iadd__(rr[l,p])

            nn2[m].finalize()
            nr2[m].finalize()
            rn2[m].finalize()
            rr2[m].finalize()

            meanlogr[:,m] = nn2[m].meanlogr
            wheta[:,m], wthetaerr[:,m] = nn2[m].calculateXi(rr[m], dr=nr2[m],rd=rn[m])
            nn_pairs[m] = nn2[m].npairs
            nn_weight[m] = nn2[m].weight

        wtheta2 = N.average(wheta, axis = 1)
        wthetaerr2 = N.average(wthetaerr, axis = 1)
        meanlogr2 = N.average(meanlogr, axis = 1)
        theta = np.exp(meanlogr2)

        cov_wheta = N.zeros((nbins,nbins))
        for l in range(nbins):
            for m in range(nbins):
                for n in range(ncen):
                    cov_wheta[l,m] += (wheta[l,n] - wtheta2[l])*(wheta[m,n] - wtheta2[m])
        cov_wheta = (ncen-1)/ncen * cov_wheta



        return theta, wheta2, cov_wheta, nn_npairs, nn_weight

    def load_tomography(self, data):

        # Columns we need from the tomography catalog
        f = self.open_input('tomography_catalog')
        source_bin = f['tomography/source_bin'][:]
        lens_bin = f['tomography/lens_bin'][:]
        f.close()

        f = self.open_input('tomography_catalog')
        r_gamma = f['multiplicative_bias/R_gamma'][:]
        f.close()

        data['source_bin']  =  source_bin
        data['lens_bin']  =  lens_bin
        data['r_gamma']  =  r_gamma

    def load_lens_catalog(self, data):
        # Subclasses can load an external lens catalog
        pass

    def load_KM_centers(self, data):
        f = self.open_input('KMcenters')
        data['KMcenters'] = f['KMcenters'][:]


    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        cat_cols = ['ra', 'dec', 'mcal_g1', 'mcal_g2', 'mcal_flags']
        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['metacal']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]

        if self.config['flip_g2']:
            data['mcal_g2'] *= -1


    def load_random_catalog(self, data):
        filename = self.get_input('random_cats')
        if filename is None:
            print("Not using randoms")
            return

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        group = f['randoms']

        cut = self.config['reduce_randoms_size']
        if 0.0<cut<1.0:
            N = group['dec'].size
            sel = np.random.uniform(size=N) < cut
        else:
            sel = slice(None)

        data['random_ra'] =  group['ra'][sel]
        data['random_dec'] = group['dec'][sel]
        data['random_e1'] =  group['e1'][sel]
        data['random_e2'] =  group['e2'][sel]
        data['random_bin'] = group['bin'][sel]

        f.close()


    def calculate_sigma_e(self, data):
        """
        Calculate sigma_e for shape catalog.
        """
        sigma_e_list = []
        mean_g1_list = []
        mean_g2_list = []
        for i in data['source_list']:
            m1, m2, mask = self.get_m(data, i)
            s = (m1+m2)/2
            g1 = data['mcal_g1'][mask]
            g2 = data['mcal_g2'][mask]
            mean_g1 = g1.mean()
            mean_g2 = g2.mean()
            # TODO Placeholder for actual weights we want to use
            w = np.ones_like(g1)
            a1 = np.sum(w**2 * (g1-mean_g1)**2)
            a2 = np.sum(w**2 * (g2-mean_g2)**2)
            b  = np.sum(w**2)
            c  = np.sum(w*s)
            d  = np.sum(w)

            sigma_e = np.sqrt( (a1/c**2 + a2/c**2) * (d**2/b) / 2. )

            sigma_e_list.append(sigma_e)
            mean_g1_list.append(mean_g1)
            mean_g2_list.append(mean_g2)

        return sigma_e_list, mean_g1_list, mean_g2_list

    def calculate_neff(self, area, data):
        neff = []
        for i in data['source_list']:
            m1, m2, mask = self.get_m(data, i)
            w    = np.ones(len(data['ra'][mask]))
            a    = np.sum(w)**2
            b    = np.sum(w**2)
            c    = area
            neff.append(a/b/c)
        return neff

    def calculate_area(self, data):
        import healpy as hp
        pix=hp.ang2pix(4096, np.pi/2.-np.radians(data['dec']),np.radians(data['ra']), nest=True)
        area=hp.nside2pixarea(4096)*(180./np.pi)**2
        mask=np.bincount(pix)>0
        area=np.sum(mask)*area
        area=float(area) * 60. * 60.
        return area

    def calculate_metadata(self, data):
        area = self.calculate_area(data)
        neff = self.calculate_neff(area, data)
        sigma_e, mean_e1, mean_e2 = self.calculate_sigma_e(data)

        meta = {}
        meta["neff"] =  neff
        meta["area"] =  area
        meta["sigma_e"] =  sigma_e
        meta["mean_e1"] =  mean_e1
        meta["mean_e2"] =  mean_e2

        return meta

class TXTwoPointLensCat(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint takes its
    lens sample from an external source instead of using
    the photometric sample.
    """
    name='TXTwoPointLensCat'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('lens_catalog', HDFFile),
    ]
    def load_lens_catalog(self, data):
        filename = self.get_input('lens_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('lens_catalog')
        data['lens_ra']  = f['lens/ra'][:]
        data['lens_dec'] = f['lens/dec'][:]
        data['lens_bin'] = f['lens/bin'][:]


class TXTwoPointPlots(PipelineStage):
    """
    Make n(z) plots

    """
    name='TXTwoPointPlots'
    inputs = [
        ('twopoint_data', SACCFile),
    ]
    outputs = [
        ('shear_xi', PNGFile),
        ('shearDensity_xi', PNGFile),
        ('density_xi', PNGFile),
    ]

    config_options = {

    }


    def run(self):
        import sacc
        import matplotlib
        matplotlib.use('agg')
        matplotlib.rcParams["xtick.direction"]='in'
        matplotlib.rcParams["ytick.direction"]='in'

        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        wtheta = sacc.standard_types.galaxy_density_xi

        filename = self.get_input('twopoint_data')
        s = sacc.Sacc.load_fits(filename)

        sources, lenses = self.read_nbin(s)
        print(f"Plotting xi for {len(sources)} sources and {len(lenses)} lenses")


        self.plot_shear_shear(s, sources)
        self.plot_shear_density(s, sources, lenses)
        self.plot_density_density(s, lenses)


    def read_nbin(self, s):
        import sacc

        xip = sacc.standard_types.galaxy_shear_xi_plus
        wtheta = sacc.standard_types.galaxy_density_xi

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(xip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        for b1, b2 in s.get_tracer_combinations(wtheta):
            lens_tracers.add(b1)
            lens_tracers.add(b2)

        sources = list(sorted(source_tracers))
        lenses = list(sorted(lens_tracers))

        return sources, lenses


    def plot_shear_shear(self, s, sources):
        import sacc
        import matplotlib.pyplot as plt

        xip = sacc.standard_types.galaxy_shear_xi_plus
        xim = sacc.standard_types.galaxy_shear_xi_minus
        nsource = len(sources)

        xi_plot = self.open_output('shear_xi', wrapper=True, figsize=(nsource*3,(nsource)*2))

        theta = s.get_tag('theta', xip)
        tmin = np.min(theta)
        tmax = np.max(theta)

        coord = lambda dt,i,j: (nsource+1-j, i) if dt==xim else (j, nsource-1-i)

        for dt in [xip, xim]:
            for i,src1 in enumerate(sources[:]):
                for j,src2 in enumerate(sources[:]):
                    D = s.get_data_points(dt, (src1,src2))


                    if len(D)==0:
                        continue

                    ax = plt.subplot2grid((nsource+2, nsource), coord(dt,i,j))

                    scale = 1e-4

                    theta = np.array([d.get_tag('theta') for d in D])
                    xi    = np.array([d.value for d in D])
                    err   = np.array([d.get_tag('error') for d  in D])
                    w = err>0
                    theta = theta[w]
                    xi = xi[w]
                    err = err[w]

                    plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.')
                    plt.xscale('log')
                    plt.ylim(-30,30)
                    plt.xlim(tmin, tmax)

                    if dt==xim:
                        if j>0:
                            ax.set_xticklabels([])
                        else:
                            plt.xlabel(r'$\theta$ (arcmin)')

                        if i==nsource-1:
                            ax.yaxis.tick_right()
                            ax.yaxis.set_label_position("right")
                            ax.set_ylabel(r'$\theta \cdot \xi_{-} \cot 10^4)$')
                        else:
                            ax.set_yticklabels([])
                    else:
                        ax.set_xticklabels([])
                        if i==nsource-1:
                            ax.set_ylabel(r'$\theta \cdot \xi_{+} \cdot 10^4$')
                        else:
                            ax.set_yticklabels([])

                    props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                    plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace=0)


        xi_plot.close()


    def plot_shear_density(self, s, sources, lenses):
        import sacc
        import matplotlib.pyplot as plt

        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        nsource = len(sources)
        nlens = len(lenses)
        xi_plot = self.open_output('shearDensity_xi', wrapper=True, figsize=(nlens*3,(nsource)*2))

        theta = s.get_tag('theta', gammat)
        tmin = np.min(theta)
        tmax = np.max(theta)


        for i,src1 in enumerate(sources):
            for j,src2 in enumerate(lenses):
                D = s.get_data_points(gammat, (src1,src2))

                if len(D)==0:
                    continue

                ax = plt.subplot2grid((nsource, nlens), (i,j))

                scale = 1e-2

                theta = np.array([d.get_tag('theta') for d in D])
                xi    = np.array([d.value for d in D])
                err   = np.array([d.get_tag('error') for d  in D])
                w = err>0
                theta = theta[w]
                xi = xi[w]
                err = err[w]

                plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.')
                plt.xscale('log')
                plt.ylim(-2,2)
                plt.xlim(tmin, tmax)

                if i==nsource-1:
                    plt.xlabel(r'$\theta$ (arcmin)')
                else:
                    ax.set_xticklabels([])

                if j==0:
                    plt.ylabel(r"$\theta \cdot \gamma_t \cdot 10^2$")
                else:
                    ax.set_yticklabels([])

                props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace=0)


        xi_plot.close()

    def plot_density_density(self, s, lenses):
        import sacc
        import matplotlib.pyplot as plt

        wtheta = sacc.standard_types.galaxy_density_xi
        nlens = len(lenses)
        xi_plot = self.open_output('density_xi', wrapper=True, figsize=(nlens*3,nlens*2))

        theta = s.get_tag('theta', wtheta)
        tmin = np.min(theta)
        tmax = np.max(theta)


        for i,src1 in enumerate(lenses[:]):
            for j,src2 in enumerate(lenses[:]):
                D = s.get_data_points(wtheta, (src1,src2))

                if len(D)==0:
                    continue

                ax = plt.subplot2grid((nlens, nlens), (i,j))

                scale = 1

                theta = np.array([d.get_tag('theta') for d in D])
                xi    = np.array([d.value for d in D])
                err   = np.array([d.get_tag('error') for d  in D])
                w = err>0
                theta = theta[w]
                xi = xi[w]
                err = err[w]

                plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.')
                plt.xscale('log')
                plt.ylim(-1,1)
                plt.xlim(tmin, tmax)

                if j>0:
                    ax.set_xticklabels([])
                else:
                    plt.xlabel(r'$\theta$ (arcmin)')

                if i==0:
                    plt.ylabel(r"$\theta \cdot w$")
                else:
                    ax.set_yticklabels([])

                props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace=0)
        xi_plot.close()



class TXGammaTFieldCenters(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of exposure fields as "lenses", as a systematics test.
    """
    name = "TXGammaTFieldCenters"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('exposures', HDFFile),
    ]
    outputs = [
        ('gammat_field_center', SACCFile),
        ('gammat_field_center_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        filename = self.get_input('exposures')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('exposures')
        data['lens_ra']  = f['exposures/ratel'][:]
        data['lens_dec'] = f['exposures/dectel'][:]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]

        fig = self.open_output('gammat_field_center_plot', wrapper=True)

        plt.errorbar(d.theta,  d.theta*d.value, d.error, fmt='ro', capsize=3)
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Field Center Tangential Shear")

        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyFieldCenter_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'fieldcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(d.value)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'fieldcenter'), d.value[i],
                theta=d.theta[i], error=d.error[i], npair=d.npair[i], weight=d.weight[i])

        #self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_field_center'), overwrite=True)

        # Also make a plot of the data

class TXGammaTBrightStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTBrightStars"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile)
    ]
    outputs = [
        ('gammat_bright_stars', SACCFile),
        ('gammat_bright_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')

        mags = f['stars/r_mag'][:]
        bright_cut = mags>14
        bright_cut &= mags<18.3

        data['lens_ra']  = f['stars/ra'][:][bright_cut]
        data['lens_dec'] = f['stars/dec'][:][bright_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]

        fig = self.open_output('gammat_bright_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (d.value - flat1) / d.error
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(d.theta) - 1)
        print('error,',d.error)

        plt.errorbar(d.theta,  d.theta*d.value, d.theta*d.error, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Bright Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(d.value)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), d.value[i],
                theta=d.theta[i], error=d.error[i], npair=d.npair[i], weight=d.weight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_bright_stars'), overwrite=True)

        # Also make a plot of the data

class TXGammaTDimStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTDimStars"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile)
    ]
    outputs = [
        ('gammat_dim_stars', SACCFile),
        ('gammat_dim_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')
        mags = f['stars/r_mag'][:]
        dim_cut = mags>18.2
        dim_cut &= mags<22

        data['lens_ra']  = f['stars/ra'][:][dim_cut]
        data['lens_dec'] = f['stars/dec'][:][dim_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]

        fig = self.open_output('gammat_dim_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (d.value - flat1) / d.error
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(d.theta) - 1)
        print('error,',d.error)

        plt.errorbar(d.theta,  d.theta*d.value, d.theta*d.error, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Dim Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(d.value)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), d.value[i],
                theta=d.theta[i], error=d.error[i], npair=d.npair[i], weight=d.weight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_dim_stars'), overwrite=True)

        # Also make a plot of the data

if __name__ == '__main__':
    PipelineStage.main()
