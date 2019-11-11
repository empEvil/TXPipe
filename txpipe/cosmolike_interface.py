from .base_stage import PipelineStage
from .data_types import HDFFile, SACCFile
import numpy as np



class TXCosmoLikeRealSpace(PipelineStage):
    name='TXCosmoLikeRealSpace'

    inputs = [
        ('tracer_metdata', HDFFile)
        ('twopoint_data', SACCFile),
    ]

    outputs = [
        ('summary_statistics', SACCFile),
    ]

    config_options = {
    # Lots of cosmolike options here, in the form:
    # name: default_value
    # or 
    # name: float/int/str/list[float]/etc.
    }


    def run(self):

        tracer_file = self.open_input('tracer_metdata')
        # Load number densities we will need
        
        # Load the sky area
        # Load the n(z) values
