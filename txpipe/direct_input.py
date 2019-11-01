from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile
from .utils.metacal import metacal_band_variants, metacal_variants
import numpy as np
from collections import namedtuple
import re

TractData = namedtuple('TractData', 
    ['shear_file', 'photo_file', 'shear_index', 'photo_index', 'star_index', 'start', 'star_start'])


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
        'metacal_directory':str,
        'object_directory':str,
    }


    def run(self):

        tracts, total_count, star_count = self.match_input_files()
        # Count the stars as well

        shear_output = photo_output = star_output = None


        if self.comm is not None and len(file_pairs) < self.comm.Get_rank():
            raise RuntimeError("Need more file pairs than MPI procs.  Because they all need some data to set up the output. Look, trust me, it's more complicated than it sounds.")

        # Can parallelize to run over these separately
        for tract in tracts:
            photo_data, shear_data = self.read_tract(tract)
            star_data = self.compute_star_data(photo_data)
            self.make_output_form(photo_data, shear_data)

            # First chunk of data we use to set up the output
            if shear_output is None:
                shear_output = self.create_output('shear_catalog', 'metacal',    shear_data, total_count)
                photo_output = self.create_output('photo_catalog', 'photometry', photo_data, total_count)
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

    def find_file_pairs(self):
        object_files = os.listdir(self.config['object_directory'])
        shear_files = os.listdir(self.config['metacal_directory'])

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
                shear_tracts[tract] = os.path.join(object_directory, f)

        pairs = []
        for tract in list(shear_tracts.keys()):
            shear_file = shear_tracts.pop(tract)
            object_file = object_tracts.pop(tract, None)
            if object_file is None:
                print(f"No matching object file for shear file {shear_file}")
                continue
            pairs.append(shear_file, object_file)
        for object_file in object_tracts.values():
            print(f"No matching shear file for object file {object_file}")

        print("Found {} pairs".format(len(pairs)))
        if not pairs:
            raise IOError("No file pairs found in specified directories matching patterns")
        return pairs




        


    def match_input_files(self):
        file_pairs = self.find_file_pairs()
        tracts = []
        start = 0
        star_start = 0
        for (shear_filename, object_filename) in file_pairs:
            shear_ids = self.read_file(shear_filename,  ['id'])['id']
            photo_data = self.read_file(object_filename, ['id', 'calib_psf_used'])
            photo_ids = photo_data['id']
            star_index = np.where(photo_data['calib_psf_used'])[0]
            shear_index, photo_index = np.intersecting_indices(shear_ids, photo_ids)
            tracts.append(TractData(shear_filename, object_filename, shear_index, photo_index, star_index, start, star_start))
            start += shear_index.size
            star_start += star_index.size
        total_count = start
        star_count = star_start
        return tracts, total_count, star_count



    def read_tract(self, tract):
        object_columns, shear_columns = self.select_input_columns()

        shear_data = self.read_file(tract.shear_filename,  shear_columns)
        photo_data = self.read_file(tract.object_filename, object_columns)

        # now sort and match the data fles by id
        for name in list(shear_data.keys()):
            shear_data[name] = shear_data[name][tract.shear_index]

        for name in list(photo_data.keys()):
            photo_data[name] = photo_data[name][tract.photo_index]
        
        return photo_data, shear_data


    def select_input_columns(self):
        bands = 'ugrizy'
        object_columns = [
            'id',
            'ra',
            'dec',
            'calib_psf_used',
            'Ixx',
            'Ixy',
            'Iyy',
            'IxxPSF',
            'IxyPSF',
            'IyyPSF',
        ]
        band_columns = [
            'mag_{}',
            'magerr_{}',
            '{}_modelfit_CModel_instFluxErr',
            '{}_modelfit_CModel_instFlux',
            '{}_mag',
            '{}_mag_err',
            'snr_{}',
        ]
        for band in bands:
            for name in band_columns:
                object_columns.append(name.format(band))

        shear_columns = (
            ['id',
            'mcal_psf_g1',
            'mcal_psf_g2',
            'mcal_psf_T_mean',
            'mcal_flags'
            ]
            + metacal_variants(
                'mcal_g1',
                'mcal_g2',
                'mcal_T',
                'mcal_s2n')
            + metacal_band_variants(bands,
                'mcal_mag',
                'mcal_mag_err')
        )

        return object_columns, shear_columns

    def read_file(self, filename, columns):
        if filename.endswith('.parquet'):
            f = pq.ParquetFile(filename)
            data = f.read(columns)
            data = {shear_data[name].to_pandas().to_numpy() for name in columns}
        else:
            raise ValueError(f"Unknown file type: {filename}")

        return data



    def compute_star_data(self, data):
        star_data = {}
        # We specifically use the stars chosen for PSF measurement
        star = data['calib_psf_used']

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
        g = output_file[group_name]
        n = len(list(data.values())[0])
        end = start + n
        for name, col in data.items():
            g[name][start:end] = col


    def create_output(self, output_tag, group, data, n):
        f = self.open_output(output_tag)
        g = f.create_group(group)
        for name, value in data.items():
            g.create_dataset(name, dtype=value.dtype, size=(n,))


# response to an old Stack Overflow question of mine:
# https://stackoverflow.com/questions/33529057/indices-that-intersect-and-sort-two-numpy-arrays
def intersecting_indices(x, y):
    u_x, u_idx_x = np.unique(x, return_index=True)
    u_y, u_idx_y = np.unique(y, return_index=True)
    i_xy = np.intersect1d(u_x, u_y, assume_unique=True)
    i_idx_x = u_idx_x[np.in1d(u_x, i_xy, assume_unique=True)]
    i_idx_y = u_idx_y[np.in1d(u_y, i_xy, assume_unique=True)]
    return i_idx_x, i_idx_y
