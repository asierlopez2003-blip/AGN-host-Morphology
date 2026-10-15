# AGN Host-Galaxy Morphology Analysis

Python pipeline for the non-parametric morphological analysis of **Active
Galactic Nucleus (AGN) host galaxies** from FITS imaging (tested on
**GTC/OSIRIS** and **CFHT/MegaCam**).

The pipeline measures the standard CAS concentration/asymmetry/smoothness
indices (**C, A, S**), the shape asymmetry (**A_s**, Pawlik et al. 2016),
**Gini** and **M20** with
[statmorph](https://github.com/iantzgirsh/statmorph), and adds an isophotal
asymmetry curve **A_x** measured at several background levels (0.5, 1, 3, 5,
9 sigma).

## Features

- **Image processing and segmentation** (`MAIN_CODE/ImageProcessing.py`)
  - Background estimation with masking and interpolation of missing pixels
  - Multi-level thresholding segmentation (`hot_snr`, `cold_snr`, `contrast`)
  - Dual hot/cold deblending with overlap control between the two detection
    levels, so the bright nuclear region and the extended host are separated
    in one map
  - Grid search over hot/cold SNR thresholds with stability checks
  - Connectivity and area gates to reject unphysical segmentation maps
  - COCO-RLE serialisation of segmentation maps
- **Morphological analysis** (`MAIN_CODE/Morphology_analysis.py`, class
  `MorphologyAnalyzer`)
  - RA/DEC → pixel coordinate conversion from FITS/WCS headers
  - Physical-size stamp extraction (e.g. 60 kpc) using the source redshift
    (Planck18 cosmology)
  - **Isophotal recentering** (optional): iterative elliptical isophote
    fitting to refine the galaxy centre before the measurement
  - **Nuclear decontamination** (optional): Gaussian-PSF subtraction scaled by
    aperture matching plus a nuclear mask (Ramos Almeida et al. 2011;
    Bonaventura et al. 2025). An adaptive gate falls back to a clean
    measurement when the decontaminated result is invalid
  - CASGM parameter extraction with statmorph (C, A, S, Gini, M20), plus A_s
  - Isophotal asymmetry curve A_x at user-defined background-σ levels
  - Automatic diagnostic figures: original stamp, segmentation map,
    isophotes, CAS panels and A_s curve

## Detector gain / weight-map protocol

Following the statmorph requirement, measurements run with the paired weight
map when available (CFHT), otherwise with the detector gain (GTC). The GTC
gain is taken from the FITS header when present; GTC OSIRIS headers do not
carry the `GAIN` keyword, so a documented fallback value is used instead.

## Repository layout

```
MAIN_CODE/          Pipeline: Morphology_analysis.py, ImageProcessing.py, PlotStyle.mplstyle
test/               Example notebooks: test_GTC.ipynb, test_CFHT.ipynb
test/data_test/     Example output figures (original, segmap, isophotal, CAS, A_s)
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12+. The notebooks additionally require Jupyter.

## Data requirements (not included)

No observational data is shipped in this repository. The pipeline expects:

- Reduced FITS images (optionally with paired weight maps, e.g.
  `*_weight_cutout_3arcmin.fits` for CFHT)
- A source catalogue with `ID`, `RA`, `DEC` and `redshift` columns

Place your files under `test/data_test/` (or edit the paths inside the
notebooks) and run the notebooks from the repository root.

## Usage

Open the example notebooks:

- `test/test_GTC.ipynb` — GTC/OSIRIS frame
- `test/test_CFHT.ipynb` — CFHT/MegaCam cutout with paired weight map

They instantiate `MorphologyAnalyzer` and call:

```python
results = analyzer.calculate_morphology_parameters(
    obj_id=...,        # object identifier
    file_path=...,     # FITS image
    ra=..., dec=...,   # sky coordinates (degrees)
    redshift=...,      # source redshift
    size_kpc=60.0,     # stamp physical size
    gain=None,         # detector gain (e-/ADU) when no weight map
    weightmap=...,     # paired weight map (if available)
    plots_dir=...,     # base output directory for diagnostic figures
)
```

returning a dictionary with the CASGM parameters, A_s, the A_x curve, the
segmentation settings and diagnostic flags, and writing the diagnostic
figures to `plots_dir/<object>/`.

## References

- Rodríguez-Gómez et al. 2019, MNRAS 483, 414 — statmorph
- Sazonova et al. 2026 — statmorph-LSST: quantifying and correcting
  morphological biases in galaxy surveys
- Pawlik et al. 2016, MNRAS 456, 1647 — shape asymmetry A_s
- Bonaventura et al. 2025, ApJ 978, 74 — Seyferts at cosmic noon are obscured
  and disturbed
- Ramos Almeida et al. 2011, MNRAS 410, 1550 — optical morphologies of the
  2 Jy radio galaxy sample

## License

MIT — see [LICENSE](LICENSE).

## Contact

Asier López Mantecón — alm223@alumnos.unican.es
