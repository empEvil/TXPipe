from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile
from .utils.metacal import metacal_band_variants, metacal_variants
import numpy as np
from collections import namedtuple
import re
import os

TractData = namedtuple('TractData', 
    ['tract', 'shear_file', 'photo_file', 'shear_index', 'photo_index', 'star_index', 'start', 'star_start'])


class TXDirectCatalogInput(PipelineStage):
    """

    """
    name='TXDirectCatalogInput'

    inputs = []

    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
        ('star_catalog', HDFFile),
    ]

    config_options = {
        'metacal_directory' :str,
        'object_directory': str,
        'metacal_zero_point':27.0,
        'metacal_bands': 'griz'
    }


    def run(self):

        if self.rank == 0:
            tracts, total_count, star_count = self.match_input_files()
            if self.comm is not None and len(tracts) < self.size:
                raise RuntimeError("Need more file pairs than MPI procs.  Because they all need some data to set up the output. Look, trust me, it's more complicated than it sounds.")
        else:
             tracts = None
             total_count = None
             star_count = None
        # Count the stars as well
        if self.comm is not None:
            tracts = self.comm.bcast(tracts)
            total_count = self.comm.bcast(total_count)
            star_count = self.comm.bcast(star_count)
        
        shear_output = photo_output = star_output = None


        # Can parallelize to run over these separately
        n = len(tracts)
        for i, tract in self.split_tasks_by_rank(enumerate(tracts)):
            print(f"Process {self.rank} reading data for tract {tract.tract} ({i+1}/{n})")
            photo_data, shear_data = self.read_tract(tract)
            self.preprocess_data(shear_data, photo_data)
            star_data = self.compute_star_data(photo_data, tract)

            # First chunk of data we use to set up the output
            if shear_output is None:
                shear_output = self.create_output('shear_catalog', 'metacal',    shear_data, total_count)
                photo_output = self.create_output('photometry_catalog', 'photometry', photo_data, total_count)
                star_output  = self.create_output('star_catalog',  'stars',       star_data, star_count)

                if self.comm is not None:
                    self.comm.Barrier()

            self.write_output(shear_output, 'metacal',    shear_data, tract.start)
            self.write_output(photo_output, 'photometry', photo_data, tract.start)
            self.write_output(star_output,  'stars',      star_data,  tract.star_start)


        if self.comm is not None:
            self.comm.Barrier()

        shear_output.close()
        photo_output.close()
        star_output.close()

    def find_tract_files(self):

        object_directory = self.config['object_directory']
        metacal_directory = self.config['metacal_directory']

        object_files = os.listdir(object_directory)
        shear_files = os.listdir(metacal_directory)

        object_tracts = {}
        shear_tracts = {}

        for f in object_files:
            m = re.match("object_tract_([0-9]+).parquet", f)
            if m:
                tract = m.group(1)
                object_tracts[tract] = os.path.join(object_directory, f)

        for f in shear_files:
            m = re.match("metacal_tract_([0-9]+).parquet", f)
            if m:
                tract = m.group(1)
                shear_tracts[tract] = os.path.join(metacal_directory, f)

        pairs = []
        for tract in list(shear_tracts.keys()):
            shear_file = shear_tracts.pop(tract)
            object_file = object_tracts.pop(tract, None)
            if object_file is None:
                print(f"No matching object file for shear file {shear_file}")
                continue
            pairs.append((tract, shear_file, object_file))
        for object_file in object_tracts.values():
            print(f"No matching shear file for object file {object_file}")

        print("Found {} pairs".format(len(pairs)))
        if not pairs:
            raise IOError("No file pairs found in specified directories matching patterns")
        return pairs




        


    def match_input_files(self):
        tract_files = self.find_tract_files()
        tracts = []
        start = 0
        star_start = 0
        print("Reading id numbers to find matches")
        n = len(tract_files)
        for i, (tract, shear_filename, object_filename) in enumerate(tract_files):
            shear_ids = self.read_file(shear_filename,  ['id'])['id']
            photo_data = self.read_file(object_filename, ['id', 'calib_psf_used'])
            if (i+1)%10==0:
                print(f" - Done matching for {i+1} of {n} file pairs")
            photo_ids = photo_data['id']
            star_index = np.where(photo_data['calib_psf_used'])[0]
            shear_index, photo_index = intersecting_indices(shear_ids, photo_ids)
            tracts.append(TractData(tract, shear_filename, object_filename, shear_index, photo_index, star_index, start, star_start))
            start += shear_index.size
            star_start += star_index.size
        total_count = start
        star_count = star_start
        return tracts, total_count, star_count



    def read_tract(self, tract):
        object_columns, shear_columns = self.select_input_columns()

        print(" - {} columns from {}".format(len(shear_columns), tract.shear_file))
        shear_data = self.read_file(tract.shear_file,  shear_columns)
        print(" - {} columns from {}".format(len(object_columns), tract.photo_file))
        photo_data = self.read_file(tract.photo_file, object_columns)


        for name in list(shear_data.keys()):
            shear_data[name] = shear_data[name][:][tract.shear_index]

        for name in list(photo_data.keys()):
            photo_data[name] = photo_data[name][:][tract.photo_index]
        
        return photo_data, shear_data


    def select_input_columns(self):
        bands = 'ugrizy'
        metacal_bands = self.config['metacal_bands']

        object_columns = [
            'id',
            'coord_ra',
            'coord_dec',
            'calib_psf_used',
            'ext_shapeHSM_HsmSourceMoments_xx',
            'ext_shapeHSM_HsmSourceMoments_xy',
            'ext_shapeHSM_HsmSourceMoments_yy',
            'base_SdssShape_psf_xx',
            'base_SdssShape_psf_xy',
            'base_SdssShape_psf_yy',
        ]
        band_columns = [
            '{}_mag',
            '{}_mag_err',
            '{}_modelfit_CModel_instFluxErr',
            '{}_modelfit_CModel_instFlux',
            '{}_modelfit_SNR',
        ]
        for b in bands:
            for name in band_columns:
                object_columns.append(name.format(b))

        shear_columns = (
            ['id',
            'mcal_psf_g1_mean',
            'mcal_psf_g2_mean',
            'mcal_psf_T_mean',
            'mcal_flags'
            ]
            + metacal_variants(
                'mcal_gauss_g1',
                'mcal_gauss_g2',
                'mcal_gauss_T',
                'mcal_gauss_s2n')
            + metacal_band_variants(metacal_bands,
                'mcal_gauss_flux',
                'mcal_gauss_flux_err')
        )

        return object_columns, shear_columns

    def read_file(self, filename, columns):
        import pyarrow.parquet as pq
        if filename.endswith('.parquet'):
            data = pq.read_table(filename, columns, use_threads=False)
            data = {name: data.column(name).to_pandas().to_numpy() for name in columns}
        else:
            raise ValueError(f"Unknown file type: {filename}")

        return data

    def preprocess_data(self, shear_data, photo_data):
        metacal_zero_point = self.config['metacal_zero_point'] 
        metacal_bands = self.config['metacal_bands']
        bands = 'ugrizy'
        
        # rename columns:
        photo_renames = {
            'coord_ra' : 'ra',
            'coord_dec' : 'dec',
            'ext_shapeHSM_HsmSourceMoments_xx' : 'Ixx',
            'ext_shapeHSM_HsmSourceMoments_xy' : 'Ixy',
            'ext_shapeHSM_HsmSourceMoments_yy' : 'Iyy',
            'base_SdssShape_psf_xx' : 'IxxPSF',
            'base_SdssShape_psf_xy' : 'IxyPSF',
            'base_SdssShape_psf_yy' : 'IyyPSF',
        }
        for b in bands:
            photo_renames[f'{b}_modelfit_SNR'] = f'snr_{b}'


        shear_renames = {
            'mcal_psf_g1_mean': 'mcal_psf_g1',
            'mcal_psf_g2_mean': 'mcal_psf_g2',
            'mcal_psf_T_mean': 'mcal_psf_T',
        }

        for v in ['', '_1p', '_2p', '_1m', '_2m']:
            shear_renames[f'mcal_gauss_g1{v}']       = f'mcal_g1{v}'
            shear_renames[f'mcal_gauss_g2{v}'] = f'mcal_g2{v}'
            shear_renames[f'mcal_gauss_T{v}']        = f'mcal_T{v}'
            shear_renames[f'mcal_gauss_s2n{v}']      = f'mcal_s2n{v}'


        for old_name, new_name in photo_renames.items():
            photo_data[new_name] = photo_data[old_name]
            del photo_data[old_name]

        for old_name, new_name in shear_renames.items():
            shear_data[new_name] = shear_data[old_name]
            del shear_data[old_name]

        # Now the magnitudes, which we have to calculate from the
        # fluxes.
        for b in metacal_bands:
            for v in ['', '_1p', '_2p', '_1m', '_2m']:
                flux = shear_data[f'mcal_gauss_flux_{b}{v}']
                err = shear_data[f'mcal_gauss_flux_err_{b}{v}']
                mag = -2.5 * np.log10(flux) + metacal_zero_point
                mag_err = (2.5 * err ) / (flux * np.log(10))

                shear_data[f'mcal_mag_{b}{v}'] = mag
                shear_data[f'mcal_mag_err_{b}{v}'] = mag_err

                del shear_data[f'mcal_gauss_flux_{b}{v}']
                del shear_data[f'mcal_gauss_flux_err_{b}{v}']

        shear_data['ra'] = photo_data['ra']
        shear_data['dec'] = photo_data['dec']
        



    def compute_star_data(self, data, tract):
        star_data = {}
        # We specifically use the stars chosen for PSF measurement
        star = tract.star_index

        # General columns
        star_data['ra'] = data['ra'][star]
        star_data['dec'] = data['dec'][star]
        star_data['id'] = data['id'][star]
        for band in 'ugrizy':
            star_data[f'{band}_mag'] = data[f'{band}_mag'][star]

        for b in 'ugrizy':
            star_data[f'{b}_mag'] = data[f'{b}_mag'][star]

        # HSM reports moments.  We convert these into
        # ellipticities.  We do this for both the star shape
        # itself and the PSF model.
        kinds = [
            ('', 'measured_'),
            ('PSF', 'model_')
        ]

        for in_name, out_name in kinds:
            # Pulling out the correct moment columns
            Ixx = data[f'Ixx{in_name}'][star]
            Iyy = data[f'Iyy{in_name}'][star]
            Ixy = data[f'Ixy{in_name}'][star]

            # Conversion of moments to e1, e2
            T = Ixx + Iyy
            e = (Ixx - Iyy + 2j * Ixy) / (Ixx + Iyy)
            e1 = e.real
            e2 = e.imag

            # save to output
            star_data[f'{out_name}e1'] = e1
            star_data[f'{out_name}e2'] = e2
            star_data[f'{out_name}T'] = T

        return star_data

    def write_output(self, output_file, group, data, start):
        g = output_file[group]
        n = len(list(data.values())[0])
        end = start + n
        print(f"Writing {group} rows {start} - {end}")
        for name, col in data.items():
            g[name][start:end] = col


    def create_output(self, output_tag, group, data, n):
        f = self.open_output(output_tag, parallel=True)
        g = f.create_group(group)
        for name, value in data.items():
            g.create_dataset(name, (n,), dtype=value.dtype)
        return f


# response to an old Stack Overflow question of mine:
# https://stackoverflow.com/questions/33529057/indices-that-intersect-and-sort-two-numpy-arrays
def intersecting_indices(x, y):
    u_x, u_idx_x = np.unique(x, return_index=True)
    u_y, u_idx_y = np.unique(y, return_index=True)
    i_xy = np.intersect1d(u_x, u_y, assume_unique=True)
    i_idx_x = u_idx_x[np.in1d(u_x, i_xy, assume_unique=True)]
    i_idx_y = u_idx_y[np.in1d(u_y, i_xy, assume_unique=True)]
    return i_idx_x, i_idx_y
