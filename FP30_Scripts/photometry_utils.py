#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import fitting, models
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.visualization import simple_norm
from ezpadova import parsec
from matplotlib.widgets import Button
from photutils import Background2D, EPSFBuilder
from photutils.background import MADStdBackgroundRMS, MMMBackground
from photutils.detection import IRAFStarFinder
from photutils.psf import (DAOGroup, IterativelySubtractedPSFPhotometry,
                           extract_stars)


__author__ = "Jonas Kemmer, @ ZAH, Landessternwarte Heidelberg"
__license__ = 'MIT'


def plot_image(image, min_max_percantage=99.):
    """Show an image normed by astropy.visualization.simple_norm.

    Parameters
    ----------
    image : 2-d Array
        Imaga data to plot.
    min_max_percantage : type
        The percentage of the image values used to determine the pixel values
        of the minimum and maximum cut levels. The lower cut level will set at
        the (100 - percent) / 2 percentile, while the upper cut level will be
        set at the (100 + percent) / 2 percentile. The default is 99.

    Returns
    -------
    figure object, plot axis
        The figure object and axis of the plot.

    """
    norm = simple_norm(image, 'log', percent=min_max_percantage)
    fig, ax = plt.subplots()
    im = ax.imshow(image, norm=norm, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax)
    fig.show()

    return fig, ax


def select_psf_stars_from_image(image):
    """Graphical selection of the stars that are used to create the PSF template.

    Parameters
    ----------
    image : 2-d Array
        Image from which the PSF stars are to be selected.
    filt : str
        Filter name.

    Returns
    -------
    list
        List of x,y-coordinates of the selected stars.

    """
    norm = simple_norm(image, 'log', percent=99.9)
    fig, ax = plt.subplots()
    ax.imshow(image, norm=norm, origin='lower')
    text = plt.text(0.5, 0.92, 'Current number of selected stars = 0\n'
                    'Double click right mouse button to exit',
                    transform=plt.gcf().transFigure, ha="center")

    points = []

    def onclick(event):
        if event.button == 1 and event.dblclick:
            points.append([int(event.xdata), int(event.ydata)])
            plt.scatter(event.xdata, event.ydata, marker='+', color='red')
            text.set_text(f'Current number of selected stars = {len(points)}\n'
                          'Double click left mouse button to select / \n'
                          'right mouse button to exit')
            fig.canvas.draw()
        if event.button == 3 and event.dblclick:
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return points


def create_psf_stamps(image, positions, psf_size):
    """Create postage stamps of the selected PSF stars.

    Parameters
    ----------
    image : 2-d Array
        Image from which the PSF stars were selected.
    positions : list
        List of x,y-coordinates of the selected stars.
    psf_size : int
        Size of the PSF stamps.

    Returns
    -------
    Tuple
        astropy.table.Table of the positional information and a list containing
        the postage stamps of the PSF stars.

    """
    psf_stamps = []
    peaks_tbl = Table(names=['x_peak', 'y_peak', 'peak_value'])
    for coord in positions:
        coarse_stamp = image[coord[1]-psf_size//2:coord[1]+psf_size//2,
                             coord[0]-psf_size//2:coord[0]+psf_size//2]
        if coarse_stamp.shape[0] == 0 or coarse_stamp.shape[1] == 0:
            continue
        center = np.argwhere(coarse_stamp == np.max(coarse_stamp))[0]
        offset = center - psf_size//2
        cent = coord + offset[::-1]
        centered_stamp = image[cent[1]-psf_size//2:cent[1]+psf_size//2,
                               cent[0]-psf_size//2:cent[0]+psf_size//2]
        if centered_stamp.shape[0] == 0 or centered_stamp.shape[1] == 0:
            continue
        psf_stamps.append(centered_stamp)
        peaks_tbl.add_row([cent[0], cent[1], np.max(centered_stamp)])
    return peaks_tbl, psf_stamps


def evaluate_selected_stars(psf_stamps, psf_size):
    """Graphical display of the selected stars for evalutation of their
    suitability as PSF stars.

    Parameters:
    ----------
    psf_stamps : list
        List containing the postage stamps of the preselected stars.
    psf_size : int
        Size of the PSF stamps.


    Returns:
    -------
    list
        List indices of the accepted stars from the input list of stamps.

    """
    center = psf_size // 2
    norm = simple_norm(psf_stamps[0], 'log', percent=99.9)
    fig, ax = plt.subplots()
    image_plot = plt.imshow(psf_stamps[0],
                            cmap='viridis',
                            interpolation='nearest',
                            origin='lower',
                            norm=norm)
    plt.scatter(center, center, marker='+', color='red')
    plt.colorbar(image_plot, label='Counts', orientation='horizontal')
    text = plt.text(0.5, 0.92, f'Showing # 1 of {len(psf_stamps)} stars.\n'
                    'Current number of selected stars = 0',
                    transform=plt.gcf().transFigure, ha="center")

    class Index(object):
        ind = 0

        selected_stars = []

        def discard_im(self, event):
            i = self.ind
            self.ind += 1
            if self.ind >= len(psf_stamps):
                plt.close()
            else:
                i = self.ind
                norm = simple_norm(psf_stamps[i], 'log', percent=99.9)
                image_plot.set_array(psf_stamps[i])
                image_plot.set_norm(norm)
                text.set_text(f'Showing # {self.ind+1} of {len(psf_stamps)} stars.\n'
                              f'Current number of selected stars = {len(self.selected_stars)}')
                fig.canvas.draw()

        def select_im(self, event):
            i = self.ind
            self.selected_stars.append(i)
            self.ind += 1
            if self.ind >= len(psf_stamps):
                plt.close()
            else:
                i = self.ind
                norm = simple_norm(psf_stamps[i], 'log', percent=99.9)
                image_plot.set_array(psf_stamps[i])
                image_plot.set_norm(norm)
                text.set_text(f'Showing # {self.ind+1} of {len(psf_stamps)} stars.\n'
                              f'Current number of selected stars = {len(self.selected_stars)}')
                fig.canvas.draw()

    callback = Index()
    axprev = plt.axes([0.3, 0.02, 0.15, 0.075])
    bprev = Button(axprev, 'Discard')
    bprev.on_clicked(callback.discard_im)
    axprev._button = bprev

    axselected = plt.axes([0.6, 0.02, 0.15, 0.075])
    bselected = Button(axselected, 'Select')
    bselected.on_clicked(callback.select_im)
    axselected._button = bselected

    plt.show()

    return callback.selected_stars


def mask_stars(peaks_tbl, selected_stars, psf_size, image_shape, filt=''):
    """Sort out stars at the edges of the image.

    Parameters
    ----------
    peaks_tbl : astropy.table.Table
        Contains the "x_peak" and "y_peak" positions of the pre-selected stars.
    selected_stars : list
        List of indices of the stars which are suitable for PSF creation.
    psf_size : int
        Size of the star-stamps and later PSF-model.
    image_shape : Tuple
        Shape of the input image.

    Returns
    -------
    astropy.table.Table object
        x and y position of the selected star, which are far enough from the
        border.

    """
    psf_stars = peaks_tbl[selected_stars]
    size = psf_size
    hsize = (size - 1) / 2
    x = psf_stars['x_peak']
    y = psf_stars['y_peak']
    mask = ((x > hsize) & (x < (image_shape[1] -1 - hsize)) &
            (y > hsize) & (y < (image_shape[0] -1 - hsize)))
    good_stars = Table()
    good_stars['x'] = x[mask]
    good_stars['y'] = y[mask]
    good_stars.write(f'selected_stars_{filt}.dat', format='ascii',
                     overwrite=True)
    return good_stars


def subtract_background(image):
    """Subtract a 2-d background from the image using
    photutils.background.MMMBackground().

    Parameters
    ----------
    image : 2-d array
        The image.

    Returns
    -------
    2-d array
        The background subtracted image.

    """
    sigma_clip = SigmaClip(sigma=5.0)
    bkg_estimator = MMMBackground()
    bkg = Background2D(image, 5, filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    return image - bkg.background


def determine_psf(image, stars_tbl, psf_size):
    """Create a PSF template from a list of given star positions in the image.

    Parameters
    ----------
    image : 2-d Array
        The image from which the stars are to be extracted.
    stars_tbl : astropy.table.Table
        Table containing the x and y positions of the stars used for creating
        the PSF template.
    psf_size : type
        Size of the PSF template.

    Returns
    -------
    photutils.psf.EPSFModel object
        A PSF model that can be used for photometry with photutils.

    """
    nddata = NDData(data=image)
    stars = extract_stars(nddata, stars_tbl, size=psf_size)
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=1,
                               progress_bar=False)
    epsf, _ = epsf_builder(stars)

    return epsf


def do_photometry(image, epsf, threshold, niters, zeropoint):
    """Performing PSF photometry with a given PSF model and threshold.
    Uses the IRAFStarFinder for source detection and DAOGroup for grouping of
    the sources.

    Parameters
    ----------
    image : 2-d array
        Image to perform photometry.
    epsf : photutils.psf.EPSFModel
        PSF used for fitting.
    threshold : int
        Multiples of the standart deviation above which the detected
        sources are fitted
    zeropoint : float
        Zeropoint magnitude of the image for flux conversion to magnitudes.


    Returns
    -------
    astropy.table.Table object
        Table with the fit results like position, flux and magnitude of the
        detected sources.

    """
    start = time.time()
    y0, x0 = np.unravel_index(np.argmax(epsf.data), epsf.data.shape)
    amp = np.max(epsf.data)
    p_init = models.Moffat2D(amp, x0, y0)
    fit_p = fitting.LevMarLSQFitter()
    yi, xi = np.indices(epsf.data.shape)
    fit_result = fit_p(p_init, xi, yi, epsf.data)
    fwhm_psf = fit_result.fwhm / 4

    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)

    iraffind = IRAFStarFinder(threshold=threshold*std,
                              fwhm=fwhm_psf,
                              minsep_fwhm=2.5,
                              roundlo=0.0, roundhi=0.2)
    fitshape = int(5*fwhm_psf)
    if fitshape % 2 == 0:
        fitshape += 1
    phot = IterativelySubtractedPSFPhotometry(group_maker=DAOGroup(2.0*fwhm_psf),
                                              bkg_estimator=MMMBackground(),
                                              psf_model=epsf,
                                              fitshape=fitshape,
                                              finder=iraffind,
                                              aperture_radius=fwhm_psf,
                                              niters=niters)
    results = phot.do_photometry(image)
    results['mag'] = zeropoint - 2.5 * np.log10(results['flux_fit'])
    results['mag_unc'] = zeropoint - 2.5 * np.log10(1 + results['flux_fit'] /
                                                    results['flux_unc'])
    residual = phot.get_residual_image()

    end = time.time()
    duration = (end - start) / 60
    print(f'Photometry was performed in {duration} minutes')
    return results, residual


def match_fitted_sources(sources_a, sources_b):
    """Compare found sources given two result tables from "do_photometry"
    and return the overlap

    Parameters
    ----------
    sources_a : astropy.table.Table
        Result table, must contain "x_fit", "y_fit" and "mag".
    sources_b : astropy.table.Table
        Result table, must contain "x_fit", "y_fit" and "mag".

    Returns
    -------
    Array
        Array of the overlapping sources, with position and magnitudes

    """
    matching_sources = []
    sources_a = np.array([sources_a['x_fit'], sources_a['y_fit'],
                          sources_a['mag']]).T
    sources_b = np.array([sources_b['x_fit'], sources_b['y_fit'],
                          sources_b['mag']]).T
    for sa, sb in product(sources_a, sources_b):
        if abs(sa[0] - sb[0]) <= 1:
            if abs(sa[1] - sb[1]) <= 1:
                matching_sources.append([sa[0], sa[1], sa[2],
                                         sb[0], sb[1], sb[2]])

    return np.array(matching_sources)


def plot_cmd(matched_sources, shift, dct_iso={}):
    """Plot a CMD with comparison isochrones.

    Parameters
    ----------
    photometry_I : astropy.table.Table
        Result table from "do_photommetry" in the I-band filter.
    photometry_V : astropy.table.Table
        Result table from "do_photommetry" in the V-band filter.
    shift : float
        Magnitude offset between the model isochrones and the observed cluster.
    dct_iso : dict
        Dictionary of the model isochrones to overplot.
        Must contain for each isochrone entry "Age", "Z" and "color".

    Returns
    -------
    Figure
        Plots the CMD with overlayed isochrones.

    """
    I = matched_sources[0]
    V = matched_sources[1]

    get_iso = lambda x, y: parsec.get_one_isochrone(x, y, phot='acs_wfc')
    vk = 'F555Wmag'
    ik = 'F814Wmag'

    plt.figure()
    for key in dct_iso.keys():
        Age = dct_iso[key]["Age"]
        if Age >= 1e9:
            Age_str = '{:.1f} Gyr'.format(float(Age)/1e9)
        else:
            Age_str = '{:.1f} Myr'.format(float(Age)/1e6)
        Z = dct_iso[key]["Z"]
        color = dct_iso[key]["color"]
        ls = dct_iso[key]["linestyle"]
        iso = get_iso(Age, Z)
        plt.plot(iso[vk][1:180].astype(float) - iso[ik][1:180].astype(float), iso[vk][1:180].astype(float) + shift, color=color,
                 linestyle=ls,
                 label='Age={a}, Z={b}'.format(
                     a=Age_str,
                     b=float(Z)))

    plt.scatter(V-I, V, s=0.08, color='black')
    plt.gca().invert_yaxis()
    plt.xlim(-1, 2)
    plt.ylim(30, 10)
    plt.legend()
    plt.xlabel('V-I')
    plt.ylabel('V')
    plt.title('Colour Magnitude Diagram BS90')
    plt.show()
