from ceci import PipelineStage
from descformats.tx import HDFFile


class TXButlerInterface1(PipelineStage):
    name='TXButlerInterface1'
    inputs = []
    outputs = [
        ('photometry_catalog', HDFFile),
    ]
    config_options = {
        'repo_dir':'/global/projecta/projectdirs/lsst/global/in2p3/Run1.1-test2/output/',
    }

    def run(self):
        # Read in some data and write it out to an HDF 5 file!
        # This gets the path from config:
        repo_dir = self.config['repo_dir']

        # This will be an h5py file open for writing:
        output_file = self.open_output('photometry_catalog')
        
        # Not sure what should come next!



if __name__ == '__main__':
    PipelineStage.main()
