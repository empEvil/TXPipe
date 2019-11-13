from .base_stage import PipelineStage
from .data_types import HDFFile, SACCFile, YamlFile
import numpy as np



class TXCosmoLikeRealSpace(PipelineStage):
    name='TXCosmoLikeRealSpace'

    # Inputs needed to calculate the covariance
    # The tracer metadata contains n_eff,
    # sigma_e, n(z), etc.
    # The fiducial cosmology is the cosmological parameters
    # The twopoint_data gives the measured correlations
    # along with the binning information.
    inputs = [
        ('tracer_metadata', HDFFile),
        ('fiducial_cosmology', YamlFile),
        ('twopoint_data', SACCFile),
    ]

    # Output is the summary stats with the covariance added
    outputs = [
        ('summary_statistics', SACCFile),
    ]

    config_options = {
    # Any cosmolike options here, in the form:
    # name: default_value
    # or 
    # name: float/int/str/list[float]/etc
    # This isn't for data that we set at earlier steps in the pipeline
    # like number of bins, or things that are calculated from input data,
    # like f_sky.  it's for other configuration options.
    }


    def run(self):
        import pyccl as ccl
        # import other stuff like cosmolike

        tracer_file = self.open_input('tracer_metadata')

        # Various scalar metadata information that we need
        metadata = tracer_file['/tracers'].attrs
        # Find out how many bins we have in the sources and lenses
        nbin_source = metadata['nbin_source']
        nbin_lens = metadata['nbin_lens']
        # Load the sky area, in sq deg
        area = metadata['area']

        # Load the z and n(z) tables for the source and lenses
        # We load a list of z_lens values, each of which is an array
        z_source = tracer_file['/tracers/n_of_z/source/z'][:]
        nz_source = [tracer_file[f'/tracers/n_of_z/source/bin_{i}'][:] for i in range(nbin_source)]
        z_lens = tracer_file['/tracers/n_of_z/lens/z'][:]
        nz_lens = [tracer_file[f'/tracers/n_of_z/lens/bin_{i}'][:] for i in range(nbin_lens)]


        # Read in the theta binning
        sacc_data = self.open_input('twopoint_data')
        # Theta angles in arcmin.  Binning is log-spaced, as in TreeCorr
        theta_min =  sacc_data.metadata['provenance/config/min_sep']
        theta_max =  sacc_data.metadata['provenance/config/max_sep']
        n_theta   =  sacc_data.metadata['provenance/config/nbins']
        

        # Read the fiducial cosmology
        cosmology_filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(cosmology_filename)
        # can now use cosmo.cosmo.params.H0 and similar to get parameters

        # Total number of data points.
        # Though we might want to cut this down first.
        N = len(sacc_data)

        # We will store each chunk of the cov mat here.
        cov_data = {}

        print("\nData in file:")
        for data_type in sacc_data.get_data_types():
            # This will codes for gamma_t, w_theta, xi_plus, xi_minus
            print(f"* {data_type}")

            for tracer_1, tracer_2 in sacc_data.get_tracer_combinations(data_type):
                # These will be the different bin pairs
                print(f"  - {tracer_1}-{tracer_2}")

                # Pull out data for this chunk, (if needed?)
                theta, xi = sacc_data.get_theta_xi(data_type, tracer_1, tracer_2)
                n = len(theta)

                # Do the work here, calling out to new processes as needed.
                # If you can put in an example of calling cosmolike I can work out the 
                # paralellization.
                cosmolike_results = ...

                # Save in dictionary for later
                cov_data[(data_type, tracer_1, tracer_2)] = cosmolike_results

        # Some work here to put the dict output in the right order
        # Matrix we will some day fill in
        C = np.zeros((N, N))

        ...

        # Attach the dovaeia
        sacc_data.add_covariance(C)

        # Now write the final output file, clobbering if needed.
        output_file = self.get_output('summary_statistics')
        sacc_data.save_fits(output_file, overwrite=True)
