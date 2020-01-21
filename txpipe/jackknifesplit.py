from .base_stage import PipelineStage
from .data_types import HDFFile, MetacalCatalog, RandomsCatalog, YamlFile, SACCFile
import numpy as np
from kmeans_radec import KMeans, kmeans_sample



class TXJackknifesplit(PipelineStage):
    name ='TXJackknifesplit'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('random_cats', RandomsCatalog),
    ]
    outputs = [
        ('Jack_labels', HDFFile),
    ]

    config_options = {
        'ncen': 10
    }

    def run(self):
        data = {}
        output_file = self.setup_output()
        self.load_shear_catalog(data)

        km = self.calculate_kmeans(data)
        labels = self.labels(data, km)
        data2 = {}
        self.load_random_catalog(data)
        labels2 = self.labels_ran(data, km)

        output_file.close()

    def calculate_kmeans(self, data):

        ra = data['ra']
        dec = data['dec']

        X = np.stack((ra,dec)).T
        km = kmeans_sample(X, self.config['ncen'], maxiter = 100)
        if not km.converged:
            km.run(X,maxiter =200)
            if not km.converged:
                print("seems like the Jackknife splitting is not converging")
        nperbin = np.bincount(km.labels)
        if np.any(nperbin == 0):
            km = kmeans_sample(X, self.config['ncen'], maxiter = 100)
            if not km.converged:
                km.run(X,maxiter =200)
                if not km.converged:
                    print("seems like the Jackknife splitting is not converging")
        return km

    def labels(self,data, km):
        ra = data['ra']
        dec = data['dec']
        X = np.stack((ra,dec)).T

        labels = km.find_nearest(X)
        return labels

    def labels_ran(self,data, km):
        ra = data['random_ra']
        dec = data['random_dec']
        X = np.stack((ra,dec)).T

        labels = km.find_nearest(X)
        return labels


    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        cat_cols = ['ra', 'dec']
        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['metacal']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]



    def load_random_catalog(self, data):
        filename = self.get_input('random_cats')
        if filename is None:
            print("Not using randoms")
            return

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','ra']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        group = f['randoms']

        #cut = self.config['reduce_randoms_size']

        #if 0.0<cut<1.0:
        #    N = group['dec'].size
        #    sel = np.random.uniform(size=N) < cut
        #else:
        sel = slice(None)

        data['random_ra'] =  group['ra'][sel]
        data['random_dec'] = group['dec'][sel]

        f.close()

    def setup_output(self):
        """
        Setting up the file that contains the jackknife regions
        """
        n = self.open_input('shear_catalog')['metacal/ra'].size
        m = self.open_input('random_cats')['randoms/ra'].size

        outfile = self.open_output('Jack_labels', parallel = True)
        group = outfile.create_group('jackknife')
        group.create_dataset('region', (n,), dtype ='i')

        #group = outfile.create_group('jackknife_random')
        group.create_dataset('region_ran', (m,), dtype ='i')

        return outfile

    def write_output(self, outfile, labels, labels_ran):

        group = outfile['jackknife']
        group['region'] = labels
        group = outfile['jackknife_random']
        group['region_ran'] = labels_ran
