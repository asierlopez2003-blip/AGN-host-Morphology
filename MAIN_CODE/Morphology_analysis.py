'''
=====================================================================================

    Morph_analysis is a python code created to analyze the morphology of astronomical
    objects using known libraries such as astropy, skimage, photutils and statmorph.

    The code processes FITS images to extract morphological parameters which gives
    important information about the structure and formation history of galaxies.

    The presented code includes the following functionalities:
    - Coordinate conversion (RA/DEC to Pixel)
    - Physical size to Pixel conversion
    - Unsharp masking
    - Morphology parameter calculation (using Statmorph LSST)

======================================================================================
'''

# ====================================================================================
#                           Import necessary libraries
# ====================================================================================

import os
import re
import sys
import time
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.visualization import simple_norm
from astropy.utils import lazyproperty
from astropy.utils.exceptions import (AstropyUserWarning, AstropyWarning)
from astropy.visualization import simple_norm, AsinhStretch, AsymmetricPercentileInterval

from scipy.ndimage import gaussian_filter, binary_dilation
from skimage import filters, img_as_float
from photutils.segmentation import detect_threshold, detect_sources

import statmorph
from statmorph.utils.image_diagnostics import make_figure

# --- Import of ImageProcessor ---
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

import ImageProcessing

# ---- Set the plotting style -----
actual_directory = os.path.dirname(os.path.abspath(__file__))
plot_style_path = os.path.join(actual_directory, 'PlotStyle.mplstyle')

if not os.path.exists(plot_style_path):
    archivos_reales = os.listdir(actual_directory)
    print(f"\n[x] ERROR DE RUTA [x]")
    print(f"Searching in: {plot_style_path}")
    print(f"Python files inside the directory: {archivos_reales}\n")
    raise FileNotFoundError("Review the list above to see the real name of the file.")

plt.style.use(plot_style_path)


# ====================================================================================
#                           MorphologyAnalyzer Class
# ====================================================================================

class MorphologyAnalyzer:
    """
    MorphologyAnalyzer is the constructed class for morphological analysis
    of astronomical objects which includes various methods.
    """

    def __init__(self):
        self.PAPER_STYLES = {
            'type1':  dict(marker='s', s=40, facecolors='none', edgecolors='#2166ac', linewidths=1.2),
            'type2':  dict(marker='o', s=40, facecolors='none', edgecolors='#b2182b', linewidths=1.2)
        }
        self.ruido_mediana = None
        self.ruido_mapa = None
        self.ruido_std = None

    def filter_detection(self, file_path):
        """
        Method for detecting the filter from the file name.
        """
        available_filters = {"U": ["U","U3"],
                             "G":["G","G3"],
                             "R":["R","R3"],
                             "I":["I","I2", "I3"],
                             "Z":["Z","Z3"]}

        file = os.path.basename(file_path)
        split_name = re.split(r'[._-]', file)

        for part in split_name:
            part_upper = part.upper()
            for main_filter, variations in available_filters.items():
                if part_upper in variations:
                    return main_filter

        print(f"\n  [!] Warning: No filters detected in: {file}\n     See if the filter is on the available_filters")
        return None

    @staticmethod
    def radec_to_pixel(header, ra, dec, origin=0):
        """
        Converts RA and DEC to pixel coordinates using WCS information from FITS header.
        """
        if isinstance(ra, str) or isinstance(dec, str):
            c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            ra_deg, dec_deg = c.ra.deg, c.dec.deg
        else:
            ra_deg, dec_deg = float(ra), float(dec)

        wcs = WCS(header)
        outputs = wcs.wcs_world2pix(np.array([ra_deg]), np.array([dec_deg]), origin)
        return float(outputs[0][0]), float(outputs[1][0])

    @staticmethod
    def physical_to_pixels(size_pc, redshift, pixel_scale_arcsec=0.254):
        """
        Function to convert the physical size of the half image (in parsecs) to pixels at a given redshift.
        """
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)
        pc_per_pixel = kpc_per_arcsec.to(u.pc / u.arcsec) * (pixel_scale_arcsec * u.arcsec / u.pixel)
        size_in_pixels = (size_pc * u.pc) / pc_per_pixel
        return size_in_pixels.value

    def crop_by_physical_size(self, image_data, header, ra, dec, size_kpc, redshift, pixel_scale=0.254):
        """
        Crops the original image based on a physical radius in kiloparsecs.
        """
        print(f" [Crop] Calculando recorte para {size_kpc} kpc a z={redshift}...")
        try:
            px, py = self.radec_to_pixel(header, ra, dec)
            center_x, center_y = int(np.round(px)), int(np.round(py))

            pixel_size_val = self.physical_to_pixels(size_kpc * 1000, redshift, pixel_scale)
            crop_radius = round(int(np.ceil(pixel_size_val))) - 1

            print(f" [Crop] Center of the AGN (x,y): ({center_x}, {center_y})")
            print(f" [Crop] Crop radius: {crop_radius} pixels. \n")

            y_min = max(0, center_y - crop_radius)
            y_max = min(image_data.shape[0], center_y + crop_radius)
            x_min = max(0, center_x - crop_radius)
            x_max = min(image_data.shape[1], center_x + crop_radius)

            if y_max <= y_min or x_max <= x_min:
                print(" [!] Error: The crop area is out of bounds or has zero size.")
                return None

            return image_data[y_min:y_max, x_min:x_max]

        except Exception as e:
            print(f" [!] Error during cropping: {e}")
            return None

    def _crop_at_center(self, image_data, center_x, center_y, crop_radius):
        """
        Crops a centred square region with the same edge logic used by
        crop_by_physical_size. Returns the cutout and the slice bounds.
        """
        y_min = max(0, int(round(center_y)) - crop_radius)
        y_max = min(image_data.shape[0], int(round(center_y)) + crop_radius)
        x_min = max(0, int(round(center_x)) - crop_radius)
        x_max = min(image_data.shape[1], int(round(center_x)) + crop_radius)
        if y_max <= y_min or x_max <= x_min:
            return None, None
        return image_data[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)

    def _decontaminate_nucleus(self, img_data, agn_px, agn_py, crop_cx, crop_cy,
                               psf_subtract=False, nuclear_mask=False,
                               psf_fwhm_px=None, psf_radius_mult=3.0,
                               nuclear_radius_mult=1.0, header=None,
                               pixel_scale=0.254):
        """
        Nuclear AGN decontamination (Phase A) applied to the background-subtracted
        image that statmorph receives:
          - psf_subtract : scale a Gaussian PSF (FWHM from the header SEEING/FWHM
            keyword, or the explicit psf_fwhm_px) to match the AGN core flux by
            least-squares aperture matching, then subtract it within
            psf_radius_mult*FWHM (Ramos Almeida et al. 2011; Bonaventura et al. 2025).
          - nuclear_mask : circular mask of nuclear_radius_mult*FWHM around the
            AGN, to be OR-ed into the statmorph segmentation mask.
        The AGN position is given in full-image pixel coordinates (agn_px,
        agn_py); the crop centre (crop_cx, crop_cy) maps it into the stamp.
        Returns (image_sub, nuclear_mask_img, diagnostics dict).
        """
        ny, nx = img_data.shape
        ax = nx // 2 + int(round(agn_px - crop_cx))
        ay = ny // 2 + int(round(agn_py - crop_cy))
        diag = {'PSF_Sub_Flag': False, 'PSF_FWHM_px': np.nan,
                'PSF_Amp_Sub': np.nan, 'PSF_Residual_Flux': np.nan,
                'Nuclear_Mask_Radius_px': np.nan}
        if not (0 <= ax < nx and 0 <= ay < ny):
            return img_data.copy(), np.zeros((ny, nx), dtype=bool), diag

        fwhm_px = psf_fwhm_px
        if fwhm_px is None and header is not None:
            seeing = header.get('SEEING', None)
            if seeing is None:
                seeing = header.get('FWHM', None)
            if seeing is not None and pixel_scale:
                fwhm_px = float(seeing) / float(pixel_scale)
        if fwhm_px is None or not np.isfinite(fwhm_px) or fwhm_px < 1.5:
            fwhm_px = 0.9 / pixel_scale
        fwhm_px = float(max(fwhm_px, 1.5))
        diag['PSF_FWHM_px'] = fwhm_px
        sigma_px = fwhm_px / 2.35482

        yy, xx = np.ogrid[:ny, :nx]
        rad = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2)
        rad2 = rad ** 2

        image_sub = img_data.copy()
        nuclear_mask_img = np.zeros((ny, nx), dtype=bool)

        if psf_subtract:
            core = rad <= fwhm_px
            core_fin = core & np.isfinite(img_data)
            if np.any(core_fin):
                enc_fraction = 1.0 - np.exp(-0.5 * (fwhm_px / sigma_px) ** 2)
                amp = np.sum(img_data, where=core_fin) / max(2.0 * np.pi * sigma_px ** 2 * enc_fraction, 1e-12)
                model = amp * np.exp(-0.5 * rad2 / sigma_px ** 2)
                sub_region = rad <= (psf_radius_mult * fwhm_px)
                finite = np.isfinite(image_sub)
                sel = sub_region & finite
                image_sub[sel] = np.maximum(image_sub[sel] - model[sel], 0.0)
                diag['PSF_Sub_Flag'] = True
                diag['PSF_Amp_Sub'] = amp
                flux_before = np.sum(img_data, where=core_fin)
                flux_after = np.sum(image_sub, where=core_fin)
                diag['PSF_Residual_Flux'] = float(flux_after / max(flux_before, 1e-12))

        if nuclear_mask:
            nuc_rad = nuclear_radius_mult * fwhm_px
            nuclear_mask_img = (rad <= nuc_rad)
            diag['Nuclear_Mask_Radius_px'] = float(nuc_rad)

        return image_sub, nuclear_mask_img, diag

    def _refine_center_ellipse(self, image, x0, y0, sma_max=None, max_iter=5, tol=0.5):
        """
        Iteratively refines the centre of the target via elliptical isophote
        fitting (photutils.isophote.Ellipse, v2 / Option C). The centre of the
        outermost converged isophote is taken as the photometric centre of the
        host galaxy. Returns (cx_ref, cy_ref, n_iter, converged).
        """
        from photutils.isophote import Ellipse, EllipseGeometry
        ny, nx = image.shape
        if sma_max is None:
            sma_max = 0.45 * min(ny, nx)
        if sma_max < 3.0:
            return float(x0), float(y0), 0, False

        cur_x, cur_y = float(x0), float(y0)
        n_iter = 0
        converged = False
        for _ in range(max_iter):
            try:
                geo = EllipseGeometry(x0=cur_x, y0=cur_y,
                                      sma=max(2.0, 0.3 * sma_max),
                                      eps=0.1, pa=0.0)
                ellipse = Ellipse(image, geometry=geo, threshold=0.1)
                isolist = ellipse.fit_image(sma0=max(1.5, 0.2 * sma_max),
                                            minsma=1.5, maxsma=sma_max,
                                            step=0.2, linear=False, fflag=0.7)
                tab = isolist.to_table()
                good = np.array(tab['stop_code']) == 0
                if not np.any(good):
                    break
                fin = np.where(good)[0][-1]
                nx_c, ny_c = float(tab['x0'][fin]), float(tab['y0'][fin])
                if not (np.isfinite(nx_c) and np.isfinite(ny_c)):
                    break
                dx, dy = nx_c - cur_x, ny_c - cur_y
                cur_x, cur_y = nx_c, ny_c
                n_iter += 1
                if np.hypot(dx, dy) < tol:
                    converged = True
                    break
            except Exception:
                break
        return cur_x, cur_y, n_iter, converged

    def unsharp_mask(self, input_path, output_path, ra, dec, sigma, radius):
        """
        Applies unsharp masking to an input FITS image and saves the resultant image.
        """
        file_name = os.path.basename(input_path)
        print(f'\n ====== Applying Unsharp Mask to {file_name} ====== \n')

        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found.")
            return

        try:
            with fits.open(input_path) as hdul:
                image_data = hdul[1].data
                header = hdul[1].header

                px, py = self.radec_to_pixel(header, ra, dec, origin=0)
                center_x, center_y = int(np.round(px)), int(np.round(py))
                crop_radius = int(radius)

                y_min = max(0, center_y - crop_radius)
                y_max = min(image_data.shape[0], center_y + crop_radius)
                x_min = max(0, center_x - crop_radius)
                x_max = min(image_data.shape[1], center_x + crop_radius)

                if y_max <= y_min or x_max <= x_min:
                    print("  Error: Crop area invalid (out of bounds or size 0).")
                    return

                cropped_image = image_data[y_min:y_max, x_min:x_max]
                image_float = img_as_float(cropped_image)

                amount = 1.0
                unsharp = filters.unsharp_mask(image_float, radius=sigma, amount=amount)

                new_header = header.copy()
                if 'CRPIX1' in new_header: new_header['CRPIX1'] -= x_min
                if 'CRPIX2' in new_header: new_header['CRPIX2'] -= y_min

                fits.PrimaryHDU(unsharp, header=new_header).writeto(output_path, overwrite=True)
                print(f"  Saved unsharp masked image to {output_path}")

                plot_output = output_path.replace('.fits', '.png')
                if plot_output == output_path: plot_output += ".png"

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(np.log10(image_float - np.min(image_float) + 1e-2), origin='lower', cmap='rainbow')
                plt.title("Original (Log)")
                plt.subplot(1, 2, 2)
                plt.imshow(np.log10(unsharp - np.min(unsharp) + 1e-2), origin='lower', cmap='rainbow')
                plt.title(f"Unsharp (sigma={sigma})")
                plt.tight_layout()
                plt.savefig(plot_output)
                plt.close()
                print(f"  Saved plot to {plot_output} \n")

        except Exception as e:
            print(f"  Error during unsharp mask processing: {e}")

    def SegmentationMap(self, image, image_smooth=None, hot_snr=30.0, cold_snr=2.0, max_iter_bg=15):
        """
        Creates an adaptive segmentation map implementing the 'Hot-Cold' parameter 
        exploration (Sazonova et al. 2026) enhanced with a topological connectivity 
        penalty: segments whose 180-rotation intersects into multiple disconnected 
        components are penalized (a local, project-specific criterion; it is NOT 
        taken from Pawlik et al. 2016 or Robotham et al. 2018).
        """
        import itertools
        import warnings
        import numpy as np
        from scipy.ndimage import label, rotate

        warnings.filterwarnings('ignore', message=".*No sources were found.*")
        warnings.filterwarnings('ignore', message=".*Cannot relabel a segmentation image.*")

        print("\n [SegmentationMap] Preparing initial image and masks...")
        img = image.copy()
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2  # Target centered at crop center

        bad_pixels_mask = np.isnan(img) | np.isinf(img)
        img[bad_pixels_mask] = 0.0
        chipmask = np.zeros_like(img, dtype=bool)

        # 1. Background Estimation
        try:
            img_limpia, bkg_median, bkg_std, bg_mask, bg_std_map = ImageProcessing.estimate_bg(
                img=img, mask=bad_pixels_mask, chipmask=chipmask,
                max_iter=max_iter_bg, thresh=0.05, fit_bg=False,
                bg_sigma=1.0, grow_sigma=3.0
            )
            self.ruido_mediana = bkg_median
            self.ruido_mapa = bg_std_map
            self.ruido_std = bkg_std
        except Exception as e:
            print(f" [x] ERROR: Error when calculating the background: {e}")
            return None

        # 2. Pre-smoothed image conditioning for detection
        if image_smooth is not None:
            img_for_detection = image_smooth.copy() - bkg_median
            img_for_detection[bad_pixels_mask] = 0.0
        else:
            img_for_detection = img_limpia.copy()

        # 3. Reference aperture for the growth-curve metric (v2)
        # A large circular aperture centred on the target approximates the total
        # flux of the host galaxy (asymptote of the flux growth curve). The map
        # selection then measures how much of this flux each candidate captures
        # per unit area, so that giant maps are penalised and compact maps that
        # already sit on the plateau of the growth curve are rewarded.
        aper_radius = 0.45 * min(ny, nx)
        yy_g, xx_g = np.ogrid[:ny, :nx]
        aper_mask = (xx_g - cx) ** 2 + (yy_g - cy) ** 2 <= aper_radius ** 2
        aper_area = aper_mask.sum()
        flujo_aper = float(np.sum(img_limpia[aper_mask]))

        # 4. Extended Parameter Space Exploration
        hot_opts = [15.0, 20.0, 30.0, 50.0]           
        cold_opts = [2.0, 2.3, 2.5, 2.7, 2.9]        
        contrast_opts = [0.001, 0.01, 0.05, 0.1, 0.5] 
        overlap_opts = [0.05, 0.10, 0.20, 0.30, 0.40]       

        parametros_grid = list(itertools.product(hot_opts, cold_opts, contrast_opts, overlap_opts))

        best_segmap, best_segmask, best_target_id = None, None, None
        best_score = -np.inf
        best_params = None

        for hot, cold, contrast, overlap in parametros_grid:
            try:
                segmask_temp, segmap_temp = ImageProcessing.get_segmap(
                    img=img_for_detection, mask=bad_pixels_mask, bgsd=bkg_std,
                    min_area=5, hot_snr=hot, cold_snr=cold,
                    nlevels=32, contrast=contrast, overlap_thresh=overlap
                )

                if segmap_temp is None or np.sum(segmask_temp) == 0:
                    continue

                labeled_data = segmap_temp.data if hasattr(segmap_temp, 'data') else segmap_temp
                if np.max(labeled_data) == 0: continue

                # Identification of Central Target ID
                central_id = labeled_data[cy, cx]
                if central_id == 0:
                    window = labeled_data[max(0, cy-4):min(ny, cy+5), max(0, cx-4):min(nx, cx+5)]
                    labels = window[window != 0]
                    if len(labels) == 0: continue
                    central_id = np.bincount(labels).argmax()

                target_mask = (labeled_data == central_id)
                area_target = np.sum(target_mask)

                # Gate A: Area Threshold (Max 35% of the 50kpc stamp)
                if area_target < 5 or (area_target / target_mask.size) > 0.35:
                    continue

                # Gate B: Positive Raw Flux Constraint
                flujo_target = np.sum(img_limpia[target_mask])
                if flujo_target <= 0:
                    continue

                # Gate C: TOPOLOGICAL CONNECTIVITY PENALTY (local criterion,
                # not from Pawlik et al. 2016)
                target_mask_rot = rotate(target_mask.astype(float), 180, reshape=False, order=0) > 0.5
                mask_intersection = target_mask & target_mask_rot
                _, num_components = label(mask_intersection)

                p_topo = 1.0 if num_components <= 1 else 0.05

                # Growth-curve capture efficiency (v2)
                # A galaxy is a finite object: its cumulative flux vs. area
                # (the growth curve) saturates. The best segmentation map relaxes
                # on the knee/plateau of the growth curve: it has already captured
                # most of the host-galaxy flux, and any extra area adds almost no
                # flux. The harmonic mean of the flux fraction and the flux density
                # (frac_flux/(1+frac_area)) peaks exactly at that knee: it is
                # penalised both when the map is too small (low flux fraction) and
                # when it is a giant map (low flux density), removing the v1 bias
                # whereby the raw (un-normalised) S/N sum always favoured giants.
                frac_flux = flujo_target / max(flujo_aper, 1e-10)
                frac_area = area_target / max(aper_area, 1)
                growth_score = frac_flux / (1.0 + frac_area)
                final_score = growth_score * p_topo

                if final_score > best_score and frac_flux > 0:
                    best_score = final_score
                    best_segmap = segmap_temp
                    best_segmask = segmask_temp
                    best_params = (hot, cold, contrast, overlap)
                    best_target_id = central_id

            except Exception:
                continue

        # Resolution & Re-labeling (Label-1 Guarantee)
        if best_segmap is not None:
            used_seg_params = {
                'Hot_SNR': best_params[0],
                'Cold_SNR': best_params[1],
                'Contrast': best_params[2],
                'Overlap': best_params[3],
                'Seg_Level_Chosen': best_params[1],
                'Growth_Score': best_score
            }
            raw_map = best_segmap.data if hasattr(best_segmap, 'data') else best_segmap
            if best_target_id != 1:
                reordered_map = np.zeros_like(raw_map, dtype=np.int32)
                reordered_map[raw_map == best_target_id] = 1
                current_lbl = 2
                for lbl in np.unique(raw_map):
                    if lbl != 0 and lbl != best_target_id:
                        reordered_map[raw_map == lbl] = current_lbl
                        current_lbl += 1
                final_map = reordered_map
            else:
                final_map = best_segmap

            self.imagen_limpia = img_limpia
            print(" [SegmentationMap] Segmentation completed successfully!")
            print(" [SegmentationMap] Used parameters:")
            for param, value in used_seg_params.items():
                print(f"    -> {param}: {value}")
            return {
                'imagen_limpia': img_limpia,
                'ruido_mapa': bg_std_map,
                'segmask_total': best_segmask,
                'segmap_final': final_map,
                'seg_params': used_seg_params
            }

        # ====================================================================
        # UNCONDITIONAL STABLE FALLBACK
        # ====================================================================
        print(" [!] WARNING: Grid search criteria not met. Reverting to STABLE baseline...")
        print(f"    -> Stable Params: Hot={hot_snr}, Cold={cold_snr}, Contrast=0.001, Overlap=0.2")
        print(" [SegmentationMap] Generating segmentation map (Hot-Cold)...")
        try:
            segmask, segmap = ImageProcessing.get_segmap(
                img=img_for_detection, mask=bad_pixels_mask, bgsd=bkg_std,
                min_area=5, hot_snr=hot_snr, cold_snr=cold_snr,
                nlevels=32, contrast=0.001, overlap_thresh=0.2
            )
        except Exception as e:
            print(f" [x] ERROR: Error when using the get_segmap function: {e}")
            return None

        if segmap is None:
            print(" [!] WARNING: No valid source was detected in the image.")
            return None

        used_seg_params = {
            'Hot_SNR': hot_snr,
            'Cold_SNR': cold_snr,
            'Contrast': 0.001,
            'Overlap': 0.2,
            'Seg_Level_Chosen': cold_snr,
            'Growth_Score': np.nan
        }

        raw_map = segmap.data if hasattr(segmap, 'data') else segmap
        central_id = raw_map[cy, cx]
        if central_id == 0:
            window = raw_map[max(0, cy-4):min(ny, cy+5), max(0, cx-4):min(nx, cx+5)]
            labels = window[window != 0]
            if len(labels) > 0:
                central_id = np.bincount(labels).argmax()
            else:
                central_id = 1

        if central_id != 1 and central_id in raw_map:
            reordered_map = np.zeros_like(raw_map, dtype=np.int32)
            reordered_map[raw_map == central_id] = 1
            current_lbl = 2
            for lbl in np.unique(raw_map):
                if lbl != 0 and lbl != central_id:
                    reordered_map[raw_map == lbl] = current_lbl
                    current_lbl += 1
            final_map = reordered_map
        else:
            final_map = segmap

        self.imagen_limpia = img_limpia
        print(" [SegmentationMap] Segmentation completed successfully!")
        return {
            'imagen_limpia': img_limpia,
            'ruido_mapa': bg_std_map,
            'segmask_total': segmask,
            'segmap_final': final_map,
            'seg_params': used_seg_params
        }
    def isophote_asymmetry(self, morph, asymmetry_isophotes=None):
        """
        Calculate isophote asymmetry following Sazonova et al. (2026).
        """
        if asymmetry_isophotes is None:
            warnings.warn('[Isophote_asymmetry] No isophote levels defined.', AstropyUserWarning)
            return -99.0

        asym_values = []
        isophote_masks = []
        for level in asymmetry_isophotes:
            isophote_mask = morph._cutout_stamp_maskzeroed >= level
            kernel_size = sorted([2, isophote_mask.shape[0]/100, 4])[1]
            kernel = Gaussian2DKernel(kernel_size)
            isophote_mask = convolve_fft(isophote_mask.astype(float), kernel, mask=morph._mask_stamp, boundary='wrap')
            isophote_mask = (isophote_mask > 0.5).astype(bool)
            isophote_masks.append(isophote_mask)

            if np.sum(isophote_mask) == 0 or np.sum(isophote_mask) < 30:
                asym_values.append(np.nan)
                continue

            asym = morph._asymmetry_function(morph._asymmetry_center, isophote_mask, 'isophote')
            asym_values.append(asym)

        self._isophotes = isophote_masks
        return asym_values

    def plot_segmentation_results(self, seg_results, plots_dir, obj_id):
        """
        Generates a 2x2 panel showing the results of the segmentation process.
        """
        if seg_results is None:
            print(" [!] WARNING: There is no data of segmentation for plotting.")
            return

        img = seg_results['imagen_limpia']
        SNR = seg_results['imagen_limpia']/seg_results['ruido_mapa']
        mascara_contaminantes = seg_results['segmask_total']
        segmap = seg_results['segmap_final']

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        norm_img = simple_norm(img, 'log', percent=99.5)

        ax = axes[0, 0]
        im1 = ax.imshow(img, origin='lower', cmap='gray', norm=norm_img)
        ax.set_title('1. Clean Imagen (Background subtracted)')
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[0, 1]
        im2 = ax.imshow(SNR, origin='lower', cmap='gray_r')
        ax.set_title('2. SNR image')
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 0]
        ax.imshow(img, origin='lower', cmap='gray', norm=norm_img)
        ax.imshow(mascara_contaminantes, origin='lower', cmap='Reds', alpha=0.3)
        ax.set_title('3. Contaminant Mask (Red)')

        ax = axes[1, 1]
        ax.imshow(img, origin='lower', cmap='gray', norm=norm_img)
        if segmap is not None:
            segmap_data = np.ma.masked_where(segmap.data == 0, segmap.data) if hasattr(segmap, 'data') else np.ma.masked_where(segmap == 0, segmap)
            ax.imshow(segmap_data, origin='lower', cmap='rainbow', alpha=0.6)
            ax.set_title('4. Final Segmentation')
        else:
            ax.set_title('4. Final Segmentation (Empty)')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{obj_id}_segmap.png'))
        plt.close()

    def plot_shape_asymmetry(self, morph, obj_id, plot_As_dir):
        """
        Function to plot the shape asymmetry segmentation map with overlaid circles.
        """
        segmap_shape = morph._segmap_shape_asym
        xca, yca = morph.xc_asymmetry, morph.yc_asymmetry
        ny, nx = segmap_shape.shape
        theta_vec = np.linspace(0, 2*np.pi, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(segmap_shape, cmap='gray', origin='lower')
        ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')

        r = morph.rpetro_circ
        ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'b', label=r'$r_{\rm petro, circ}$')
        r = morph.rpetro_ellip
        ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r', label=r'$r_{\rm petro, ellip}$')
        r = morph.rmax_circ
        ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'c', lw=1.5, label=r'$r_{\rm max}$')

        text = (r'$A_S = %.4f$' % (morph.shape_asymmetry,))
        ax.text(0.05, 0.05, text, fontsize=12, horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))

        ax.set_title(f'Shape Asymmetry Segmap for {obj_id}', fontsize=14)
        ax.legend(loc=4, fontsize=10, facecolor='w', framealpha=1.0, edgecolor='k')
        plt.xlim(-0.5, nx - 0.5); plt.ylim(-0.5, ny - 0.5)

        as_figs_dir = os.path.join(plot_As_dir, 'A_s_Figs')
        os.makedirs(as_figs_dir, exist_ok=True)
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        fig.savefig(os.path.join(as_figs_dir, f'{obj_id}_As.png'))
        plt.close()

    def plot_isophotes(self, obj_id, asymmetry_isophotes, morph, plot_iso_dir):
        im = morph._cutout_stamp_maskzeroed.copy()
        ny, nx = im.shape
        im = AsinhStretch(0.3)(AsymmetricPercentileInterval(0.5, 99.5)(im))

        plt.figure(figsize=(6, 6))
        plt.imshow(im, cmap='gray', origin='lower')
        if hasattr(self, '_isophotes') and self._isophotes is not None and asymmetry_isophotes is not None:
            leyenda_elementos = []
            for i, iso in enumerate(self._isophotes):
                color = plt.get_cmap('plasma_r')(i / len(self._isophotes))
                valor_nivel = asymmetry_isophotes[i]
                etiqueta = f"Back. std: {valor_nivel:.2f}" if isinstance(valor_nivel, float) else f"Level: {valor_nivel:.4f}"
                plt.contour(iso, levels=[0.9, 1.1], colors=[color], linewidths=1)
                linea_proxy = plt.Line2D([0], [0], color=color, linewidth=1, label=etiqueta)
                leyenda_elementos.append(linea_proxy)
            plt.legend(handles=leyenda_elementos, loc='best', framealpha=0.7, facecolor='white')

        plt.xlim(-0.5, nx - 0.5); plt.ylim(-0.5, ny - 0.5)
        plt.title(r'Isophotal Asymmetry, $A_X$', fontsize=14)
        plt.gca().get_xaxis().set_visible(False); plt.gca().get_yaxis().set_visible(False)
        plt.savefig(os.path.join(plot_iso_dir, f'{obj_id}_isophotal.png'))
        plt.close()

    def calculate_morphology_parameters(self, obj_id, file_path, ra, dec, redshift, size_kpc, plots_dir, gain=None, weightmap=None, isophote_level=[0.1, 0.2, 0.3], pixel_scale=0.254, recenter=False, psf_subtract=False, nuclear_mask=False, psf_fwhm_px=None, psf_radius_mult=3.0, nuclear_radius_mult=1.0):
        """
        Calculates morphology parameters for a single object using statmorph.
        Applies Gaussian smoothing to the full original FITS image before cropping
        (Sazonova et al. 2026, Appendix C), delivering both raw and smoothed
        cutouts to the segmentation pipeline.

        Parameters
        ----------
        recenter : bool, default False
            When True, the crop centre (initially the astrometric RA/DEC position)
            is iteratively refined via elliptical isophote fitting on the
            background-subtracted, pre-smoothed image (Option C). The stamp is
            re-cropped at the photometric centre of the host galaxy so that the
            subsequent morphological measurements (C, A, S, A_s, A_x) are
            centred on the galaxy rather than the AGN position.
        psf_subtract : bool, default False
            When True, the unresolved AGN nucleus is removed from the
            background-subtracted image before running statmorph by scaling a
            Gaussian PSF (FWHM from the header SEEING/FWHM keyword) to the AGN
            core flux and subtracting it within psf_radius_mult*FWHM (Phase A;
            Ramos Almeida et al. 2011; Bonaventura et al. 2025).
        nuclear_mask : bool, default False
            When True, a circular mask of nuclear_radius_mult*FWHM around the
            AGN is OR-ed into the segmentation mask so the masked nuclear region
            is excluded from the statmorph measurements.
        psf_fwhm_px : float, optional
            Explicit PSF FWHM in pixels. When None, it is derived from the
            SEEING/FWHM header keyword and pixel_scale; otherwise a typical
            ground-based seeing of 0.9 arcsec is assumed.
        psf_radius_mult : float, default 3.0
            Radius (in units of the PSF FWHM) inside which the scaled PSF is
            subtracted from the image.
        nuclear_radius_mult : float, default 1.0
            Radius (in units of the PSF FWHM) of the circular nuclear mask.
        """
        analyzed_path = os.path.join(plots_dir, 'Analyzed')
        treated_path = os.path.join(plots_dir, 'Original')
        segmap_path = os.path.join(plots_dir, 'Segmaps')
        isophotal_path = os.path.join(plots_dir, 'Isophotals')

        for p in [analyzed_path, treated_path, segmap_path, isophotal_path]:
            os.makedirs(p, exist_ok=True)

        print(f'\n >>>>>>>>>>>>>>>> PROCESSING MORPHOLOGY FOR {obj_id} <<<<<<<<<<<<<<<< \n')
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return None

        try:
            with fits.open(file_path) as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                     image, header = hdul[1].data.copy(), hdul[1].header
                elif len(hdul) > 0 and hdul[0].data is not None:
                     image, header = hdul[0].data.copy(), hdul[0].header
                else:
                    print(f"\n [x]  ERROR: No valid image data found in {file_path}")
                    return None

                # ====================================================================
                # MANDATORY FULL-IMAGE PRE-SMOOTHING BEFORE CROPPING (Sazonova et al. 2026, App. C)
                # ====================================================================
                print("  ----> Applying pre-smoothing (sigma=1.5px) to original full FITS image for detection (Sazonova et al. 2026, App. C)...")
                image_smoothed = gaussian_filter(image, sigma=1.5)

                # Crop BOTH the raw image (for Statmorph) and smoothed image (for Photutils detection) (Sazonova et al. 2026, App. C)
                cropped_image = self.crop_by_physical_size(image, header, ra, dec, size_kpc, redshift, pixel_scale)
                cropped_image_smooth = self.crop_by_physical_size(image_smoothed, header, ra, dec, size_kpc, redshift, pixel_scale)

                if cropped_image is None or cropped_image_smooth is None:
                    print("\n  [×] Error: Image cutting has failed. Leaving the analysis ... \n")
                    return None

                px, py = self.radec_to_pixel(header, ra, dec)
                center_x, center_y = int(np.round(px)), int(np.round(py))
                agn_px, agn_py = center_x, center_y
                crop_radius = round(int(np.ceil(self.physical_to_pixels(size_kpc * 1000, redshift, pixel_scale)))) - 1

                y_min, y_max = max(0, center_y - crop_radius), min(image.shape[0], center_y + crop_radius)
                x_min, x_max = max(0, center_x - crop_radius), min(image.shape[1], center_x + crop_radius)

                # ====================================================================
                # ACTIVE ISOPHOTAL RE-CENTERING (Option C / v2) — (Sazonova et al. 2026)
                # Refine the crop centre from the astrometric AGN position to the
                # photometric centre of the host galaxy using iterative elliptical
                # isophote fitting (photutils.isophote.Ellipse). Re-crop once at the
                # refined centre so that subsequent morphological measurements (C,
                # A, S, A_s, A_x) are centred on the galaxy, not the AGN.
                # ====================================================================
                cent_off_x, cent_off_y = 0.0, 0.0
                num_cent_iter = 0
                if recenter and cropped_image_smooth is not None:
                    print("  [Recenter] Refining stamp center via isophotal fitting...")
                    # Estimate background on the smoothed crop for a clean isophote fit
                    bad_px = np.isnan(cropped_image_smooth) | np.isinf(cropped_image_smooth)
                    chipmask = np.zeros_like(cropped_image_smooth, dtype=bool)
                    try:
                        img_limpia_rc, _, _, _, _ = ImageProcessing.estimate_bg(
                            img=cropped_image_smooth.copy(), mask=bad_px, chipmask=chipmask,
                            max_iter=15, thresh=0.05, fit_bg=False, bg_sigma=1.0, grow_sigma=3.0)
                    except Exception:
                        img_limpia_rc = cropped_image_smooth.copy() - np.nanmedian(cropped_image_smooth)
                    ny_rc, nx_rc = img_limpia_rc.shape
                    sma_max_rc = 0.45 * min(ny_rc, nx_rc)
                    x0_rc, y0_rc = nx_rc // 2.0, ny_rc // 2.0
                    for _ in range(5):
                        ref_x, ref_y, nfit, ok = self._refine_center_ellipse(
                            img_limpia_rc, x0_rc, y0_rc, sma_max=sma_max_rc,
                            max_iter=5, tol=0.5)
                        dx, dy = ref_x - x0_rc, ref_y - y0_rc
                        cent_off_x += dx
                        cent_off_y += dy
                        num_cent_iter += nfit
                        x0_rc, y0_rc = x0_rc + dx, y0_rc + dy
                        if np.hypot(dx, dy) < 0.5:
                            break
                    print(f"  [Recenter] Offset: ({cent_off_x:+.2f}, {cent_off_y:+.2f}) px "
                          f"in {num_cent_iter} isophote fit(s)")
                    # Re-crop centered on the refined position
                    new_cx = center_x + int(round(cent_off_x))
                    new_cy = center_y + int(round(cent_off_y))
                    crop_re, bounds = self._crop_at_center(image, new_cx, new_cy, crop_radius)
                    crop_sm_re, bounds_sm = self._crop_at_center(
                        image_smoothed, new_cx, new_cy, crop_radius)
                    if crop_re is not None and crop_sm_re is not None:
                        cropped_image, cropped_image_smooth = crop_re, crop_sm_re
                        center_x, center_y = new_cx, new_cy
                        y_min, y_max, x_min, x_max = bounds
                        print(f"  [Recenter] Re-cropped at refined centre ({new_cx}, {new_cy}).")
                    else:
                        print("  [!] Recentered crop out of bounds; keeping WCS center.")
                        cent_off_x, cent_off_y = 0.0, 0.0
                        num_cent_iter = 0

                plt.figure()
                plt.imshow(np.log10(cropped_image - np.min(cropped_image) + 1e-2), origin='lower', cmap='gray_r')
                plt.title(f"Cropped {obj_id}")
                plt.savefig(os.path.join(treated_path, f'{obj_id}_original.png'))
                plt.close()

                kernel = Gaussian2DKernel(2); kernel.normalize()
                psf = kernel.array

                # ===== Segmentation Map =====
                # Pass BOTH raw and smoothed cutouts to the parameter grid explorer (Sazonova et al. 2026, App. C)
                segmap_result = self.SegmentationMap(image=cropped_image, image_smooth=cropped_image_smooth, hot_snr=30.0, cold_snr=2.0)
                if segmap_result is None:
                    print("\n [×] ERROR: Error when creating the segmentation map.")
                    return None

                # Extract the segmentation parameters used for this image
                seg_params = segmap_result.get('seg_params', {})

                segmap_matrix = np.array(segmap_result['segmap_final'].data if hasattr(segmap_result['segmap_final'], 'data') else segmap_result['segmap_final'], dtype=np.int32)
                segmask_matrix = np.array(segmap_result['segmask_total'].data if hasattr(segmap_result['segmask_total'], 'data') else segmap_result['segmask_total'], dtype=bool)

                self.plot_segmentation_results(segmap_result, segmap_path, obj_id=obj_id)

                # ===== STATMORPH =====
                start_time = time.time()
                img_data = np.array(segmap_result['imagen_limpia'].data if hasattr(segmap_result['imagen_limpia'], 'data') else segmap_result['imagen_limpia'], dtype=np.float64)

                # ============================================================================
                # NUCLEAR AGN DECONTAMINATION (Phase A)
                # Subtract the AGN point source (scaled PSF) and/or mask the nuclear
                # region BEFORE running statmorph, so that C, A, S, A_s and Gini/M20
                # measure the host galaxy rather than the AGN nucleus.
                # ============================================================================
                if psf_subtract or nuclear_mask:
                    img_data_raw = img_data.copy()
                    img_data, nuc_mask, nuc_diag = self._decontaminate_nucleus(
                        img_data, agn_px, agn_py, center_x, center_y,
                        psf_subtract=psf_subtract, nuclear_mask=nuclear_mask,
                        psf_fwhm_px=psf_fwhm_px, psf_radius_mult=psf_radius_mult,
                        nuclear_radius_mult=nuclear_radius_mult,
                        header=header, pixel_scale=pixel_scale)
                    if nuc_mask.any():
                        segmask_matrix = segmask_matrix | nuc_mask
                    self._last_decontamination = {
                        'img_before': img_data_raw,
                        'img_after': img_data.copy(),
                        'mask_after': segmask_matrix.copy(),
                        'diag': dict(nuc_diag)
                    }
                    print("  [Decontamination] PSF_Sub:", nuc_diag['PSF_Sub_Flag'],
                          "| FWHM:", f"{nuc_diag['PSF_FWHM_px']:.2f}px",
                          "| Amp:", f"{nuc_diag['PSF_Amp_Sub']:.3e}",
                          "| Core_resid_frac:", f"{nuc_diag['PSF_Residual_Flux']:.3f}",
                          "| Nuc_mask_r:", f"{nuc_diag['Nuclear_Mask_Radius_px']:.1f}px")
                else:
                    nuc_diag = {'PSF_Sub_Flag': False, 'PSF_FWHM_px': np.nan,
                                'PSF_Amp_Sub': np.nan, 'PSF_Residual_Flux': np.nan,
                                'Nuclear_Mask_Radius_px': np.nan}

                if weightmap is not None:
                    with fits.open(weightmap) as w_hdul:
                        weightmap_data = w_hdul[0].data.copy()
                    weightmap = np.array(weightmap_data[y_min:y_max, x_min:x_max], dtype=np.float64)
                    print("\n [✓] Weightmap loaded successfully. \n Running statmorph with weightmap ...")
                    source_morphs = statmorph.source_morphology(
                        img_data, segmap=segmap_matrix, mask=segmask_matrix,
                        weightmap=weightmap, psf=psf, include_doublesersic=False
                    )
                elif gain is not None:
                    print("\n [✓] No weightmap provided.\n Running statmorph with detector gain ...")
                    source_morphs = statmorph.source_morphology(
                        img_data, segmap=segmap_matrix, mask=segmask_matrix,
                        gain=gain, psf=psf, include_doublesersic=False
                    )
                else:
                    print("\n  [!] WARNING: No weightmap or gain provided. \n Exiting without morphology calculation ... \n")
                    return None

                if len(source_morphs) > 0:
                    morph = source_morphs[0]

                    # CATASTROPHIC FAILURE SAFEGUARD (statmorph flag == 4)
                    if morph.flag == 4:
                        print(f"\n  [!] STATMORPH WARNING: Object morphology is catastrophic (Flag 4). Skipping to avoid crash.")
                        return None

                    isophote_level = [self.ruido_std*0.5, self.ruido_std*1.0, self.ruido_std*3.0, self.ruido_std*5.0, self.ruido_std*9.0]
                    isophote_asym_values = self.isophote_asymmetry(morph=morph, asymmetry_isophotes=isophote_level)
                    print(f"\n  Isophote asymmetry values for level:\n")
                    for i, level in enumerate(isophote_level):
                        print(f"    Back. std Level = {level:<6.4f} |  A_x = {isophote_asym_values[i]:.4f}")

                    self.plot_isophotes(obj_id=obj_id, asymmetry_isophotes=isophote_level, morph=morph, plot_iso_dir=isophotal_path)
                    print('\n  ----> Statmorph running time: %g s.' % (time.time() - start_time), '\n')

                    fig = make_figure(morph)
                    fig.savefig(os.path.join(analyzed_path, f'{obj_id}_analysis.png'))
                    plt.close(fig)

                    self.plot_shape_asymmetry(morph, obj_id, analyzed_path)

                    # 1. Definimos la lista base del diccionario
                    results = {
                        'Object_ID': obj_id,
                        'Redshift': redshift,
                        'Concentration (C)': morph.concentration,
                        'Asymmetry (A)': morph.asymmetry,
                        'Smoothness (S)': morph.smoothness,
                        'Shape_Asymmetry (A_s)': morph.shape_asymmetry,
                        'Gini': morph.gini,
                        'M20': morph.m20,
                        'Flag': morph.flag,
                        # Parámetros de segmentación (Correctamente implementados)
                        'Hot_SNR': seg_params.get('Hot_SNR', np.nan),
                        'Cold_SNR': seg_params.get('Cold_SNR', np.nan),
                        'Contrast': seg_params.get('Contrast', np.nan),
                        'Overlap': seg_params.get('Overlap', np.nan),
                        # Diagnostic columns (v2): recentering offset and growth-curve level
                        'Cent_Offset_X': cent_off_x,
                        'Cent_Offset_Y': cent_off_y,
                        'Num_Isophote_Iter': num_cent_iter,
                        'Seg_Level_Chosen': seg_params.get('Cold_SNR', np.nan),
                        # Phase A: nuclear AGN decontamination diagnostics
                        'PSF_Sub_Flag': nuc_diag['PSF_Sub_Flag'],
                        'PSF_FWHM_px': nuc_diag['PSF_FWHM_px'],
                        'PSF_Amp_Sub': nuc_diag['PSF_Amp_Sub'],
                        'PSF_Residual_Flux': nuc_diag['PSF_Residual_Flux'],
                        'Nuclear_Mask_Radius_px': nuc_diag['Nuclear_Mask_Radius_px']
                    }

                    # 2. Desempaquetamos A_X en columnas numéricas independientes
                    levels_names = ['0.5std', '1.0std', '3.0std', '5.0std', '9.0std']
                    if isinstance(isophote_asym_values, list):
                        for name_lvl, val in zip(levels_names, isophote_asym_values):
                            results[f'A_x_{name_lvl}'] = val
                    else:
                        for name_lvl in levels_names:
                            results[f'A_x_{name_lvl}'] = np.nan

                    print(f"\n  [✓]  Morphology parameters calculated successfully\n ")
                    for name, value in results.items():
                        if name == 'Object_ID': print(f"\n=========== {value} MORPHOLOGICAL RESULTS ===========\n")
                        else: print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
                    return results
                else:
                    print("  Statmorph found no valid sources.")
                    return None

        except Exception as e:
            print(f"\n [×] ERROR: Error processing object {obj_id}: {e}")
            return None
