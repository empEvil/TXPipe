from .base_stage import PipelineStage
from .data_types import NOfZFile, HDFFile, DiagnosticMaps, TomographyCatalog
import numpy as np

class TXTracerMetaData(PipelineStage):


    name='TXTracerMetadata'
    inputs = [
        ('photoz_stack', NOfZFile),
        ('tomography_catalog', TomographyCatalog),
        ('diagnostic_maps', DiagnosticMaps),
    ]
    outputs = [
        ('tracer_metadata', HDFFile),        
    ]
    config_options = {
    }


    def run(self):
        import h5py

        area = self.read_area()

        # Input and output files
        tomo_file = self.open_input('tomography_catalog')
        pz_file   = self.open_input('photoz_stack')
        meta_file = self.open_output('tracer_metadata')

        # We only want to copy some pieces of the tomography data
        # (the small stuff and some attributes)
        self.copy_data(tomo_file, meta_file, 'multiplicative_bias', 'tracers', 'R_gamma_mean')
        self.copy_data(tomo_file, meta_file, 'multiplicative_bias', 'tracers', 'R_S')
        self.copy_data(tomo_file, meta_file, 'multiplicative_bias', 'tracers', 'R_total')
        self.copy_data(tomo_file, meta_file, 'tomography', 'tracers', 'N_eff')
        self.copy_data(tomo_file, meta_file, 'tomography', 'tracers', 'lens_counts')
        self.copy_data(tomo_file, meta_file, 'tomography', 'tracers', 'sigma_e')
        self.copy_data(tomo_file, meta_file, 'tomography', 'tracers', 'source_counts')
        self.copy_number_density(tomo_file, meta_file, area)
        self.copy_attributes(tomo_file, meta_file, 'tomography', 'tracers')

        # The n(z) data is small anyway, so we copy it in its entirety
        pz_file.copy('n_of_z', meta_file['tracers/'], 'n_of_z')

        # Close up the metadata file so it can be renamed
        # to its final destination
        meta_file.close()

    def read_area(self):
        # Read the area of the unmasked map from the map input file
        map_file = self.open_input("diagnostic_maps")
        area = map_file['maps'].attrs['area']

        # Check that the unit is sq deg, as we expect.
        # Otherwise we need to code a conversion
        area_unit = map_file['maps'].attrs['area_unit']
        if area_unit != 'sq deg':
            raise ValueError("Expected area to be in sq deg")

        return area


    # Utility function to copy a data set from one HDF5 file to anotther
    def copy_data(self, in_file, out_file, in_section, out_section, name):
        x = in_file[f'{in_section}/{name}'][:]
        out_file.create_dataset(f'{out_section}/{name}', data=x)


    # Utility function to copy attributes from one HDF5 file to anotther
    def copy_attributes(self, in_file, out_file, name, out_name):
        for k,v in in_file[name].attrs.items():
            out_file[out_name].attrs[k] = v

    # Copy the number density data in particular into the new file.
    def copy_number_density(self, tomo_file, meta_file, area):
        # Load the count information and use it to get the density
        # Effective and raw source counts, and lens count
        N_eff = tomo_file['tomography/N_eff'][:]
        source_counts = tomo_file['tomography/source_counts'][:]
        lens_counts = tomo_file['tomography/lens_counts'][:]


        # Convert to densities
        area_sq_arcmin = area * 60**2

        n_eff = N_eff / area_sq_arcmin
        lens_density = lens_counts / area_sq_arcmin
        source_density = source_counts / area_sq_arcmin

        # ANd save everything to the output
        meta_file.create_dataset('tracers/n_eff', data=n_eff)
        meta_file.create_dataset('tracers/lens_density', data=lens_density)
        meta_file.create_dataset('tracers/source_density', data=source_density)
        meta_file['tracers'].attrs['area'] = area
        meta_file['tracers'].attrs['area_unit'] = 'sq deg'
