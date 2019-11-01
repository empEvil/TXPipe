from .base_stage import PipelineStage
from .data_types import TomographyCatalog, HDFFile, SACCFile
from .utils import NumberDensityStats
from .utils.metacal import metacal_variants, metacal_band_variants
import numpy as np
import warnings
import glob

class TXTruthRedshift(PipelineStage):
    name='TXTruthRedshift'


    inputs = [
        ('photometry_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
        ('twopoint_data', SACCFile),
    ]

    outputs = [
        ('twopoint_data_true_z', SACCFile),
        ('true_redshift_catalog', HDFFile),
    ]

    config_options = {
        "match_catalog_root": "/global/projecta/projectdirs/lsst/groups/SSim/DC2/matched_ids_dc2_object_run2.1i_dr4",
        "zmax": 4.0,
        "nz": 400,
        "chunk_rows": 200_000,
    }

    def run(self):
        import fitsio, sacc
        # Config for the output n(z)
        zmin = 0.0
        zmax = self.config['zmax']
        nz = self.config['nz']
        dz = zmax / nz
        z = np.arange(zmin, zmax, dz)

        # Input data for matching and result size
        lookup_table = self.load_match()
        nbin_source, nbin_lens, N = self.read_counts()

        #  Data we will generate
        redshift_outfile = self.open_output('true_redshift_catalog')
        redshift_outfile.create_dataset('redshift_true/z', (N,), dtype=np.int64)
        nz_source = [np.zeros(nz) for i in range(nbin_source)]
        nz_lens = [np.zeros(nz) for i in range(nbin_lens)]

        # Might be running in parallel so break here just to sync
        if self.comm:
            self.comm.Barrier()


        # Make the iterators to read the input data
        chunk_rows = self.config['chunk_rows']
        it_photo = self.iterate_hdf('photometry_catalog', 'photometry', ['id'], chunk_rows)
        it_tomo = self.iterate_hdf('tomography_catalog', 'tomography', ['source_bin', 'lens_bin'], chunk_rows)


        for (s,e, photo_data), (_, _, tomo_data) in zip(it_photo, it_tomo):
            print(f"Processing rows {s}-{e}")
            #  lookup truth values
            true_z = self.lookup(photo_data['id'], lookup_table)

            # Save truth values
            redshift_outfile['redshift_true/z'][s:e] = true_z

            # Build up n(z)
            for i in range(nbin_source):
                w = np.where(tomo_data['source_bin']==i)
                count, _ = np.histogram(true_z[w], bins=nz, range=(zmin, zmax))
                nz_source[i] += count
            # Same for lens
            for i in range(nbin_lens):
                w = np.where(tomo_data['lens_bin']==i)
                count, _ = np.histogram(true_z[w], bins=nz, range=(zmin, zmax))
                nz_lens[i] += count

        self.write_output(z, nz_source, nz_lens)
                
    def write_output(self, z, nz_source, nz_lens):
        import sacc
        # Load the input sacc data as a template
        S = sacc.Sacc.load_fits(self.get_input('twopoint_data'))

        # Replace the n(z) data in it
        for i,nz in enumerate(nz_source):
            t = S.tracers[f'source_{i}']
            t.z = z
            t.nz = nz

        for i,nz in enumerate(nz_lens):
            t = S.tracers[f'lens_{i}']
            t.z = z
            t.nz = nz

        S.metadata['redshift_is_true'] = True

        # And save.  The metadata should be maintained
        output_file = self.get_output('twopoint_data_true_z')
        S.save_fits(output_file)


    def read_counts(self):
        # Some basic numbers we need from the input file
        photo = self.open_input('photometry_catalog')
        tomo = self.open_input('tomography_catalog')        
        nbin_source = tomo['tomography'].attrs['nbin_source']
        nbin_lens = tomo['tomography'].attrs['nbin_lens']
        N = tomo['tomography/lens_bin'].size
        photo.close()
        tomo.close()
        return nbin_source, nbin_lens, N



    def load_match(self):
        import fitsio
        pattern = self.config['match_catalog_root'] + "*"
        files = glob.glob(pattern)

        ids = []
        redshift = []

        for fn in files:
            print(f"Loading match catalog {fn}")
            f = fitsio.FITS(fn)
            d=f[1].read_columns(['objectId','redshift_true'])
            
            redshift.append(d['redshift_true'])
            ids.append(d['objectId'])

        ids = np.concatenate(ids)
        redshift = np.concatenate(redshift)
            
        a = np.argsort(ids)
        ids = ids[a]
        redshift = redshift[a]

        return ids, redshift


    def lookup(self, object_ids, match_data):
        # look up objectId values in relevant tract
        ids, redshift = match_data

        # for each id in object_ids, find in ids and get redshift.
        # if not found, use 0? nan?
        indices = np.searchsorted(ids, object_ids)
        z = redshift[indices]
        m = ids[indices] == object_ids
        f = np.sum(m) / m.size
        print(f"{f:.2%} of objects matched")
        return z
