'''

Test script for MorphologyAnalyzer class for unsharp masking and morphology parameter calculation.

'''

# Import necessary libraries
from Morph_analysis import MorphologyAnalyzer
import os
import numpy as np
from astropy.io import fits

# Initialize the MorphologyAnalyzer
analyzer = MorphologyAnalyzer()

# Load object information from FITS file

... # 'ID', 'redshift', 'RA', 'DEC'  

# Example of Unsharp Mask
analyzer.unsharp_mask(
        input_path="...", # input file path
        output_path="out.fits", # output file path
        ra=..., # RA of the object
        dec=..., # DEC of the object
        sigma=..., # Sigma for Gaussian blur 
        radius=... # size of the image to make the unsharp mask on in pixels
)


# Example Morphology Parameter Calculation
results = analyzer.calculate_morphology_parameters(
    obj_id="ID", # ID of the object
    file_path="out.fits", # We compute the morphology on the unsharp masked image
    ra=..., # RA of the object
    dec=..., # DEC of the object
    redshift=..., # redshift of the object
    size_kpc=..., # size of the image to analyze in kpc
    gain = ..., # gain of the instrument
    output_dir_base="..." # base output directory
)

