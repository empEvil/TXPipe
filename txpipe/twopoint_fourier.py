from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps, HDFFile, PhotozPDFFile
import numpy as np
import collections
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator
import sys
import warnings


# Using the same convention as in twopoint.py
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2



NAMES = {SHEAR_SHEAR:"shear-shear", SHEAR_POS:"shear-position", POS_POS:"position-position"}

Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'l', 'value', 'win', 'i', 'j'])

class TXTwoPointFourier(PipelineStage):
    """This Pipeline Stage computes all auto- and cross-correlations
    for a list of tomographic bins, including all galaxy-galaxy,
    galaxy-shear and shear-shear power spectra. Sources and lenses
    both come from the same shear_catalog and tomography_catalog objects.

    The power spectra are computed after deprojecting a number of
    systematic-dominated modes, represented as input maps.

    In the future we will want to include the following generalizations:
     - TODO: specify which cross-correlations in particular to include
             (e.g. which bin pairs and which source/lens combinations).
     - TODO: include flags for rejected objects. Is this included in
             the tomography_catalog?
     - TODO: ell-binning is currently static.
    """
    name = 'TXTwoPointFourier'
    inputs = [
        ('photoz_stack', HDFFile),  # Photoz stack
        ('diagnostic_maps', DiagnosticMaps),
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('tomography_catalog', TomographyCatalog),  # For density info
    ]
    outputs = [
        ('twopoint_data_fourier', SACCFile)
    ]

    config_options = {
        "mask_threshold": 0.0,
        "bandwidth": 0,
        "apodization_size": 3.0,
        "apodization_type": "C1",  #"C1", "C2", or "Smooth"
        "flip_g2": False,
    }

    def run(self):
        import pymaster
        import healpy
        import sacc
        config = self.config

        if self.comm:
            self.comm.Barrier()

        self.setup_results()


        # Generate namaster fields
        pixel_scheme, f_d, f_wl, nbin_source, nbin_lens, f_sky = self.load_maps()
        if self.rank==0:
            print("Loaded maps and converted to NaMaster fields")

        # Get the complete list of calculations to be done,
        # for all the three spectra and all the bin pairs.
        # This will be used for parallelization.
        calcs = self.select_calculations(nbin_source, nbin_lens)


        # Load in the per-bin auto-correlation noise levels and 
        # mean response values
        noise, mean_R = self.load_tomographic_quantities(nbin_source, nbin_lens, f_sky)

        # Load the n(z) values, which are both saved in the output
        # file alongside the spectra, and used to calcualate the
        # fiducial theory C_ell
        tracers = self.load_tracers(nbin_source, nbin_lens)

        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins = self.choose_ell_bins(pixel_scheme, f_sky)

        # This is needed in the deprojection calculation
        theory_cl = self.fiducial_theory(tracers, ell_bins, nbin_source, nbin_lens)

        # If we are rank zero print out some info
        if self.rank==0:
            nband = ell_bins.get_n_bands()
            ell_effs = ell_bins.get_effective_ells()
            print(f"Chosen {nband} ell bin bands with effective ell values and ranges:")
            for i in range(nband):
                leff = ell_effs[i]
                lmin = ell_bins.get_ell_min(i)
                lmax = ell_bins.get_ell_max(i)
                print(f"    {leff:.0f}    ({lmin:.0f} - {lmax:.0f})")




        # Namaster uses workspaces, which we re-use between
        # bins
        w00, w02, w22 = self.setup_workspaces(pixel_scheme, f_d, f_wl, ell_bins)
        print(f"Rank {self.rank} set up workspaces")

        # Run the compute power spectra portion in parallel
        # This splits the calculations among the parallel bins
        # It's not the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment.
        for i, j, k in self.split_tasks_by_rank(calcs):
            self.compute_power_spectra(
                pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, noise, theory_cl)

        if self.rank==0:
            print(f"Collecting results together")
        # Pull all the results together to the master process.
        self.collect_results()

        # Write the collect results out to HDF5.
        if self.rank == 0:
            self.save_power_spectra(tracers, nbin_source, nbin_lens)
            print("Saved power spectra")




    def load_maps(self):
        import pymaster as nmt
        import healpy
        # Parameters that we need at this point
        apod_size = self.config['apodization_size']
        apod_type = self.config['apodization_type']


        # Load the various input maps and their metadata
        map_file = self.open_input('diagnostic_maps', wrapper=True)
        pix_info = map_file.read_map_info('mask')
        area = map_file.file['maps'].attrs["area"]

        nbin_source = map_file.file['maps'].attrs['nbin_source']
        nbin_lens = map_file.file['maps'].attrs['nbin_lens']

        # Choose pixelization and read mask and systematics maps
        pixel_scheme = choose_pixelization(**pix_info)

        # Load the mask. It should automatically be the same shape as the
        # others, based on how it was originally generated.
        # We remove any pixels that are at or below our threshold (default=0)
        mask = map_file.read_map('mask')
        mask_threshold = self.config['mask_threshold']
        mask[mask <= mask_threshold] = 0
        mask[np.isnan(mask)] = 0
        mask_sum = mask.sum()
        f_sky = area / 41253.
        if self.rank == 0:
            print(f"Unmasked area = {area}, fsky = {f_sky}")


        # Load all the maps in.
        # TODO: make it possible to just do a subset of these
        ngal_maps = [map_file.read_map(f'ngal_{b}') for b in range(nbin_lens)]
        g1_maps = [map_file.read_map(f'g1_{b}') for b in range(nbin_source)]
        g2_maps = [map_file.read_map(f'g2_{b}') for b in range(nbin_source)]

        # Mask any pixels which have the healpix bad value
        for m in g1_maps + g2_maps + ngal_maps:
            mask[m==healpy.UNSEEN] = 0

        if self.config['flip_g2']:
            for g2 in g2_maps:
                w = np.where(g2!=healpy.UNSEEN)
                g2[w]*=-1

        # TODO: load systematics maps here, once we are making them.
        syst_nc = None
        syst_wl = None

        map_file.close()

        # Cut out any pixels below the threshold,
        # zeroing out any pixels there
        cut = mask < mask_threshold
        mask[cut] = 0

        # We also apply this cut to the count maps,
        # since otherwise the pixels below threshold would contaminate
        # the mean calculation below.
        for ng in ngal_maps:
            ng[cut] = 0


        # Convert the number count maps to overdensity maps.
        # First compute the overall mean object count per bin.
        # Maybe we should do this in the mapping code itself?
        n_means = [ng.sum()/mask_sum for ng in ngal_maps]
        # and then use that to convert to overdensity
        d_maps = [(ng/mu)-1 for (ng,mu) in zip(ngal_maps, n_means)]

        d_fields = []
        wl_fields = []

        if pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.size_x)
            ly = np.radians(pixel_scheme.size_y)

            # Apodize the mask
            if apod_size > 0:
                if self.rank==0:
                    print(f"Apodizing mask with size {apod_size} deg and method {apod_type}")
                mask = nmt.mask_apodization_flat(mask, lx, ly, apod_size, apotype=apod_type)
            elif self.rank==0:
                print("Not apodizing mask.")

            for i,d in enumerate(d_maps):
                # Density for gnomonic maps
                if self.rank == 0:
                    print(f"Generating gnomonic density field {i}")
                field = nmt.NmtFieldFlat(lx, ly, mask, [d], templates=syst_nc) 
                d_fields.append(field)

            for i,(g1,g2) in enumerate(zip(g1_maps, g2_maps)):
                # Density for gnomonic maps
                if self.rank == 0:
                    print(f"Generating gnomonic lensing field {i}")
                field = nmt.NmtFieldFlat(lx, ly, mask, [g1,g2], templates=syst_wl) 
                wl_fields.append(field)
            

        elif pixel_scheme.name == 'healpix':
            # Apodize the mask
            if apod_size > 0:
                if self.rank==0:
                    print(f"Apodizing mask with size {apod_size} deg and method {apod_type}")
                mask = nmt.mask_apodization(mask, apod_size, apotype=apod_type)
            elif self.rank==0:
                print("Not apodizing mask.")


            for i,d in enumerate(d_maps):
                # Density for gnomonic maps
                print(f"Generating healpix density field {i}")
                field = nmt.NmtField(mask, [d], templates=syst_nc)
                d_fields.append(field)

            for i,(g1,g2) in enumerate(zip(g1_maps, g2_maps)):
                # Density for gnomonic maps
                print(f"Generating healpix lensing field {i}")
                field = nmt.NmtField(mask, [g1, g2], templates=syst_wl) 
                wl_fields.append(field)

        else:
            raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

        return pixel_scheme, d_fields, wl_fields, nbin_source, nbin_lens, f_sky


    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank == 0:
            # Concatenate results
            self.results = sum(self.results, [])

    def setup_results(self):
        self.results = []

    def choose_ell_bins(self, pixel_scheme, f_sky):
        import pymaster as nmt
        from .utils.nmt_utils import MyNmtBinFlat, MyNmtBin
        if pixel_scheme.name == 'healpix':
            # This is just approximate
            area = f_sky * 4 * np.pi
            width = np.sqrt(area) #radians
            nlb = self.config['bandwidth']
            nlb = nlb if nlb>0 else max(1,int(2 * np.pi / width))
            ell_bins = MyNmtBin(int(pixel_scheme.nside), nlb=nlb)
        elif pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.nx * pixel_scheme.pixel_size_x)
            ly = np.radians(pixel_scheme.ny * pixel_scheme.pixel_size_y)
            ell_min = max(2 * np.pi / lx, 2 * np.pi / ly)
            ell_max = min(pixel_scheme.nx * np.pi / lx, pixel_scheme.ny * np.pi / ly)
            d_ell = self.config['bandwidth']
            d_ell = d_ell if d_ell>0 else 2 * ell_min
            n_ell = int((ell_max - ell_min) / d_ell) - 1
            l_bpw = np.zeros([2, n_ell])
            l_bpw[0, :] = ell_min + np.arange(n_ell) * d_ell
            l_bpw[1, :] = l_bpw[0, :] + d_ell
            ell_bins = MyNmtBinFlat(l_bpw[0, :], l_bpw[1, :])
            ell_bins.ell_mins = l_bpw[0, :]
            ell_bins.ell_maxs = l_bpw[1, :]

        return ell_bins

    def setup_workspaces(self, pixel_scheme, f_d, f_wl, ell_bins):
        import pymaster as nmt
        # choose scheme class
        if pixel_scheme.name == 'healpix':
            workspace_class = nmt.NmtWorkspace
        elif pixel_scheme.name == 'gnomonic':
            workspace_class = nmt.NmtWorkspaceFlat
        else:
            raise ValueError(f"No NaMaster workspace for pixel scheme {pixel_scheme.name}")

        # Compute mode-coupling matrix
        # TODO: mode-coupling could be pre-computed and provided in config.
        w00 = workspace_class()
        w00.compute_coupling_matrix(f_d[0], f_d[0], ell_bins)
        if self.rank==0:
            print("Computed w00 coupling matrix")

        w02 = workspace_class()
        w02.compute_coupling_matrix(f_d[0], f_wl[0], ell_bins)
        if self.rank==0:
            print("Computed w02 coupling matrix")

        w22 = workspace_class()
        w22.compute_coupling_matrix(f_wl[0], f_wl[0], ell_bins)
        if self.rank==0:
            print("Computed w22 coupling matrix")

        return w00, w02, w22

    def select_calculations(self, nbins_source, nbins_lens):
        calcs = []

        # For shear-shear we omit pairs with j<i
        k = SHEAR_SHEAR
        for i in range(nbins_source):
            for j in range(i + 1):
                calcs.append((i, j, k))

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in range(nbins_source):
            for j in range(nbins_lens):
                calcs.append((i, j, k))

        # For position-position we omit pairs with j<i
        k = POS_POS
        for i in range(nbins_lens):
            for j in range(i + 1):
                calcs.append((i, j, k))

        return calcs

    def compute_power_spectra(self, pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, noise, cl_theory):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        # TODO: Fix window functions, and how to save them.

        # k refers to the type of measurement we are making
        import sacc
        import pymaster
        CEE=sacc.standard_types.galaxy_shear_cl_ee
        CBB=sacc.standard_types.galaxy_shear_cl_bb
        CdE=sacc.standard_types.galaxy_shearDensity_cl_e
        CdB=sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd=sacc.standard_types.galaxy_density_cl

        type_name = NAMES[k]
        print(f"Process {self.rank} calcluating {type_name} spectrum for bin pair {i},{i}")
        sys.stdout.flush()

        # We need the theory spectrum for this pair
        theory = cl_theory[(i,j,k)]

        # The binning information - effective (mid) ell values and
        # the window information
        ls = ell_bins.get_effective_ells()
        win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]

        if k == SHEAR_SHEAR:
            workspace = w22
            field_i = f_wl[i]
            field_j = f_wl[j]
            cl_guess = [theory, np.zeros_like(theory), np.zeros_like(theory), np.zeros_like(theory)]
            results_to_use = [(0, CEE), (3, CBB)]

        elif k == POS_POS:
            workspace = w00
            field_i = f_d[i]
            field_j = f_d[j]
            cl_guess = [theory]
            results_to_use = [(0, Cdd)]

        elif k == SHEAR_POS:
            workspace = w02
            field_i = f_wl[i]
            field_j = f_d[j]
            cl_guess = [theory, np.zeros_like(theory)]
            results_to_use = [(0, CdE), (1, CdB)]

        cl_noise = self.compute_noise(i,j,k,ell_bins,workspace,noise)

        if pixel_scheme.name == 'healpix':
            c = pymaster.compute_full_master(field_i, field_j, ell_bins,
                cl_noise=cl_noise, cl_guess=cl_guess, workspace=workspace)
        elif pixel_scheme.name == 'gnomonic':
            c = pymaster.compute_full_master_flat(field_i, field_j, ell_bins,
                cl_noise=cl_noise, cl_guess=cl_guess, ells_guess=cl_theory['ell'],
                workspace=workspace)

        for index, name in results_to_use:
            self.results.append(Measurement(name, ls, c[index], win, i, j))


    def compute_noise(self, i, j, k, ell_bins, w, noise):
        # No noise contribution from cross-correlations
        if (i!=j) or (k==SHEAR_POS):
            return None
        # print("Setting noise to zero")
        # return None
        # We loaded in sigma_e and the densities
        # earlier on and put them in the noise dictionary
        noise_level = noise[(i,k)]

        if ell_bins.is_flat():
            n = ell_bins.get_n_bands()
        else:
            n = w.wsp.lmax + 1

        # ell-by-ell noise level of the right size
        N1 = np.ones(n) * noise[(i,k)]

        # # Need the same noise for EE, BB, EB, BE
        # # or PE, PB
        if k==SHEAR_SHEAR:
            N2 = [N1, N1, N1, N1]
        elif k==SHEAR_POS:
            N2 = [N1, N1]
        else:
            N2 = [N1]

        if ell_bins.is_flat():
            ell = ell_bins.get_effective_ells()
            N3 = w.couple_cell(ell, N2)
        else:
            N3 = w.couple_cell(N2)
        return N3


    def load_tomographic_quantities(self, nbin_source, nbin_lens, f_sky):
        tomo = self.open_input('tomography_catalog')
        sigma_e = tomo['tomography/sigma_e'][:]
        mean_R = tomo['multiplicative_bias/R_gamma_mean'][:]
        N_eff = tomo['tomography/N_eff'][:]
        lens_counts = tomo['tomography/lens_counts'][:]
        tomo.close()

        area = 4*np.pi*f_sky

        n_eff = N_eff / area
        n_lens = lens_counts / area

        noise = {}
        for i in range(nbin_source):
            noise[(i,SHEAR_SHEAR)] = f_sky*sigma_e[i]**2 / n_eff[i]
        for i in range(nbin_lens):
            noise[(i,POS_POS)] = f_sky / n_lens[i]

        warnings.warn("Using unweighted lens samples here")

        return noise, mean_R



    def load_tracers(self, nbin_source, nbin_lens):
        import sacc
        f = self.open_input('photoz_stack')

        tracers = {}

        for i in range(nbin_source):
            name = f"source_{i}"
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            T = sacc.BaseTracer.make("NZ", name, z, Nz)
            tracers[name] = T

        for i in range(nbin_lens):
            name = f"lens_{i}"
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            T = sacc.BaseTracer.make("NZ", name, z, Nz)
            tracers[name] = T

        return tracers

    def fiducial_theory(self, tracers, ell_bins, nbin_source, nbin_lens):
        import pyccl as ccl

        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        # We will need the theory C_ell in a continuum up until
        # the full ell_max, because we will need a weighted sum
        # over the values
        ell_max = np.max(ell_bins.ell_max)
        #f_d[0].fl.lmax
        ell = np.arange(ell_max+1, dtype=int)

        # Convert from SACC tracers (which just store N(z))
        # to CCL tracers (which also have cosmology info in them).
        CTracers = {}

        # Lensing tracers - need to think a little more about
        # the fiducial intrinsic alignment here
        for i in range(nbin_source):
            x = tracers[f'source_{i}']
            tag = ('S', i)
            CTracers[tag] = ccl.WeakLensingTracer(cosmo, dndz=(x.z, x.nz))
        # Position tracers - even more important to think about fiducial biases
        # here - these will be very very wrong otherwise!
        # Important enough that I'll put in a warning.
        warnings.warn("Not using galaxy bias in fiducial theory density spectra")

        for i in range(nbin_lens):
            x = tracers[f'lens_{i}']
            tag = ('P', i) 
            b = np.ones_like(x.z)
            CTracers[tag] = ccl.NumberCountsTracer(cosmo, dndz=(x.z, x.nz),
                                        has_rsd=False, bias=(x.z,b))

        # Use CCL to actually calculate the C_ell values for the different cases
        theory_cl = {}
        theory_cl['ell'] = ell
        k = SHEAR_SHEAR
        for i in range(nbin_source):
            for j in range(i+1):
                Ti = CTracers[('S',i)]
                Tj = CTracers[('S',j)]
                # The full theory C_ell over the range 0..ellmax
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)
                

        # The same for the galaxy galaxy-lensing cross-correlation
        k = SHEAR_POS
        for i in range(nbin_source):
            for j in range(nbin_lens):
                Ti = CTracers[('S',i)]
                Tj = CTracers[('P',j)]
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

        # And finally for the density correlations
        k = POS_POS
        for i in range(nbin_lens):
            for j in range(i+1):
                Ti = CTracers[('P',i)]
                Tj = CTracers[('P',j)]
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

        return theory_cl


    def save_power_spectra(self, tracers, nbin_source, nbin_lens):
        import sacc
        from sacc.windows import TopHatWindow
        CEE=sacc.standard_types.galaxy_shear_cl_ee
        CBB=sacc.standard_types.galaxy_shear_cl_bb
        CdE=sacc.standard_types.galaxy_shearDensity_cl_e
        CdB=sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd=sacc.standard_types.galaxy_density_cl

        S = sacc.Sacc()

        for tracer in tracers.values():
            S.add_tracer_object(tracer)

        for d in self.results:
            tracer1 = f'source_{d.i}' if d.corr_type in [CEE, CBB, CdE, CdB] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [CEE, CBB] else f'lens_{d.j}'

            n = len(d.l)
            for i in range(n):
                win = TopHatWindow(d.win[i][0], d.win[i][1])
                S.add_data_point(d.corr_type, (tracer1, tracer2), d.value[i], ell=d.l[i], window=win, i=d.i, j=d.j)

        # Save provenance information
        for key, value in self.gather_provenance().items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    S.metadata[f'provenance/{key}_{i}'] = v
            else:
                S.metadata[f'provenance/{key}'] = value



        output_filename = self.get_output("twopoint_data_fourier")
        S.save_fits(output_filename, overwrite=True)


if __name__ == '__main__':
    PipelineStage.main()


