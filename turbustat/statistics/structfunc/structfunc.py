# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division
import numpy as np
import scipy

from astropy import units as u
from astropy.wcs import WCS
from copy import copy
import statsmodels.api as sm
from warnings import warn
from astropy.utils.console import ProgressBar

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..stats_utils import common_scale, padwithzeros
from ..fitting_utils import check_fit_limits, residual_bootstrap
from ..stats_warnings import TurbuStatMetricWarning
from ..lm_seg import Lm_Seg
from ..convolve_wrapper import convolution_wrapper

class StructFunc2d(BaseStatisticMixIn):
    '''

    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, order=2,
                 lags=None, nlags=25, distance=None, beam=None):
        super(StructFunc2d, self).__init__()

        # Set the data and perform checks
        self.input_data_header(img, header)

        if weights is None:
            # self.weights = np.ones(self.data.shape)
            self.weights = np.isfinite(self.data).astype(float)
        else:
            self.weights = input_data(weights, no_header=True)

        if distance is not None:
            self.distance = distance

        if lags is None:
            min_size = 3.0
            self.lags = \
                np.logspace(np.log10(min_size),
                            np.log10(min(self.data.shape) / 2.), nlags) * u.pix
        else:
            # Check if the given lags are a Quantity
            # Default to pixel scales if it isn't
            if not hasattr(lags, "value"):
                self.lags = lags * u.pix
            else:
                self.lags = self._to_pixel(lags)

        self.order = order
        self.load_beam(beam=beam)

    @property
    def lags(self):
        '''
        Lag values.
        '''
        return self._lags

    @lags.setter
    def lags(self, values):

        if not isinstance(values, u.Quantity):
            raise TypeError("lags must be given as an astropy.units.Quantity.")

        pix_lags = self._to_pixel(values)

        if np.any(pix_lags.value < 1):
            raise ValueError("At least one of the lags is smaller than one "
                             "pixel. Remove these lags from the array.")

        # Catch floating point issues in comparing to half the image shape
        half_comp = (np.floor(pix_lags.value) - min(self.data.shape) / 2.)

        if np.any(half_comp > 1e-10):
            raise ValueError("At least one of the lags is larger than half of"
                             " the image size. Remove these lags from the "
                             "array.")

        self._lags = values

    @property
    def weights(self):
        '''
        Array of weights.
        '''
        return self._weights

    @weights.setter
    def weights(self, arr):

        if arr.shape != self.data.shape:
            raise ValueError("Given weight array does not match the shape of "
                             "the given image.")

        self._weights = arr

    def compute_structfunc2d(self, boundary='wrap',
                             show_progress=True):
        '''

        '''

        self._structfunc = np.empty((len(self.lags)))
        self._structfunc_error = np.empty((len(self.lags)))

        if show_progress:
            bar = ProgressBar(len(self.lags))

        for i, lag in enumerate(self.lags.value):
            print(i, lag)

        if show_progress:
            bar.update(i + 1)

    @property
    def structfunc(self):
        '''
        structure function values.
        '''
        return self._structfunc

    @property
    def structfunc_error(self):
        '''
        1-sigma errors on the structure function values.
        '''
        return self._structfunc_error

    def fit_plaw(self, xlow=None, xhigh=None, brk=None, verbose=False,
                 bootstrap=False, bootstrap_kwargs={},
                 **fit_kwargs):
        '''
        Fit a power-law to the structure function.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        bootstrap : bool, optional
            Bootstrap using the model residuals to estimate the standard
            errors.
        bootstrap_kwargs : dict, optional
            Pass keyword arguments to `~turbustat.statistics.fitting_utils.residual_bootstrap`.
        verbose : bool, optional
            Show fit summary when enabled.
        '''

        x = np.log10(self.lags.value)
        y = np.log10(self.structfunc)

        if xlow is not None:
            xlow = self._to_pixel(xlow)

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.structfunc, dtype=bool)
            xlow = self.lags.min() * 0.99

        if xhigh is not None:
            xhigh = self._to_pixel(xhigh)

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.structfunc, dtype=bool)
            xhigh = self.lags.max() * 1.01

        self._fit_range = [xlow, xhigh]

        within_limits = np.logical_and(lower_limit, upper_limit)

        y = y[within_limits]
        x = x[within_limits]

        weights = self.structfunc_error[within_limits] ** -2

        min_fits_pts = 3

        if brk is not None:
            # Try fitting a segmented model

            pix_brk = self._to_pixel(brk)

            if pix_brk < xlow or pix_brk > xhigh:
                raise ValueError("brk must be within xlow and xhigh.")

            model = Lm_Seg(x, y, np.log10(pix_brk.value), weights=weights)

            fit_kwargs['verbose'] = verbose
            fit_kwargs['cov_type'] = 'HC3'

            model.fit_model(**fit_kwargs)

            self.fit = model.fit

            if model.params.size == 5:

                # Check to make sure this leaves enough to fit to.
                if sum(x < model.brk) < min_fits_pts:
                    warn("Not enough points to fit to." +
                         " Ignoring break.")

                    self._brk = None
                else:
                    good_pts = x.copy() < model.brk
                    x = x[good_pts]
                    y = y[good_pts]

                    self._brk = 10**model.brk * u.pix

                    self._slope = model.slopes

                    if bootstrap:
                        stderrs = residual_bootstrap(model.fit,
                                                     **bootstrap_kwargs)

                        self._slope_err = stderrs[1:-1]
                        self._brk_err = np.log(10) * self.brk.value * \
                            stderrs[-1] * u.pix

                    else:
                        self._slope_err = model.slope_errs
                        self._brk_err = np.log(10) * self.brk.value * \
                            model.brk_err * u.pix

                    self.fit = model.fit

            else:
                self._brk = None
                # Break fit failed, revert to normal model
                warn("Model with break failed, reverting to model\
                      without break.")
        else:
            self._brk = None

        # Revert to model without break if none is given, or if the segmented
        # model failed.
        if self.brk is None:

            x = sm.add_constant(x)

            # model = sm.OLS(y, x, missing='drop')
            model = sm.WLS(y, x, missing='drop', weights=weights)

            self.fit = model.fit(cov_type='HC3')

            self._slope = self.fit.params[1]

            if bootstrap:
                stderrs = residual_bootstrap(self.fit,
                                             **bootstrap_kwargs)
                self._slope_err = stderrs[1]

            else:
                self._slope_err = self.fit.bse[1]

        self._bootstrap_flag = bootstrap

        if verbose:
            print(self.fit.summary())

            if self._bootstrap_flag:
                print("Bootstrapping used to find stderrs! "
                      "Errors may not equal those shown above.")

        self._model = model

    @property
    def brk(self):
        '''
        Fitted break point.
        '''
        return self._brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point in the segmented linear model.
        '''
        return self._brk_err

    @property
    def slope(self):
        '''
        Fitted slope.
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        Standard error on the fitted slope.
        '''
        return self._slope_err

    @property
    def fit_range(self):
        '''
        Range of lags used in the fit.
        '''
        return self._fit_range

    def fitted_model(self, xvals):
        '''
        Computes the fitted power-law in log-log space using the
        given x values.

        Parameters
        ----------
        xvals : `~numpy.ndarray`
            Values of log(lags) to compute the model at (base 10 log).

        Returns
        -------
        model_values : `~numpy.ndarray`
            Values of the model at the given values.
        '''

        if isinstance(self._model, Lm_Seg):
            return self._model.model(xvals)
        else:
            return self.fit.params[0] + self.fit.params[1] * xvals

    def plot_fit(self, save_name=None, xunit=u.pix, symbol='o', color='r',
                 fit_color='k', label=None,
                 show_residual=True):
        '''
        Plot the structure function curve and the fit.

        Parameters
        ----------
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        symbol : str, optional
            Shape to plot the data points with.
        color : {str, RGB tuple}, optional
            Color to show the structure function curve in.
        fit_color : {str, RGB tuple}, optional
            Color of the fitted line. Defaults to `color` when no input is
            given.
        label : str, optional
            Label to later be used in a legend.
        show_residual : bool, optional
            Plot the fit residuals.
        '''

        if fit_color is None:
            fit_color = color

        import matplotlib.pyplot as plt

        fig = plt.gcf()
        axes = plt.gcf().get_axes()
        if len(axes) == 0:
            if show_residual:
                ax = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
                ax_r = plt.subplot2grid((4, 1), (3, 0), colspan=1,
                                        rowspan=1,
                                        sharex=ax)
            else:
                ax = plt.subplot(111)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            ax = axes[0]
            ax_r = axes[1]

        ax.set_xscale("log")
        ax.set_yscale("log")

        lags = self._spatial_unit_conversion(self.lags, xunit).value

        # Check for NaNs
        fin_vals = np.logical_or(np.isfinite(self.structfunc),
                                 np.isfinite(self.structfunc_error))
        ax.errorbar(lags[fin_vals], self.structfunc[fin_vals],
                    yerr=self.structfunc_error[fin_vals],
                    fmt="{}-".format(symbol), color=color,
                    label=label, zorder=-1)

        xvals = np.linspace(self._fit_range[0].value,
                            self._fit_range[1].value,
                            100) * self.lags.unit
        xvals_conv = self._spatial_unit_conversion(xvals, xunit).value

        ax.plot(xvals_conv, 10**self.fitted_model(np.log10(xvals.value)),
                '--', color=fit_color, linewidth=2)

        xlow = \
            self._spatial_unit_conversion(self._fit_range[0], xunit).value
        xhigh = \
            self._spatial_unit_conversion(self._fit_range[1], xunit).value

        ax.axvline(xlow, color=color, alpha=0.5, linestyle='-.')
        ax.axvline(xhigh, color=color, alpha=0.5, linestyle='-.')

        # ax.legend(loc='best')
        ax.grid(True)

        if show_residual:
            resids = self.structfunc - 10**self.fitted_model(np.log10(lags))
            ax_r.errorbar(lags[fin_vals], resids[fin_vals],
                          yerr=self.structfunc_error[fin_vals],
                          fmt="{}-".format(symbol), color=color,
                          zorder=-1)

            ax_r.set_ylabel("Residuals")

            ax_r.set_xlabel("Lag ({})".format(xunit))

            ax_r.axhline(0., color=fit_color, linestyle='--')

            ax_r.axvline(xlow, color=color, alpha=0.5, linestyle='-.')
            ax_r.axvline(xhigh, color=color, alpha=0.5, linestyle='-.')
            ax_r.grid()

            plt.setp(ax.get_xticklabels(), visible=False)

        else:
            ax.set_xlabel("Lag ({})".format(xunit))

        ax.set_ylabel(r"$S_{}$".format(self.order))

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, verbose=False, xunit=u.pix,
            boundary='wrap',
            xlow=None, xhigh=None,
            brk=None, fit_kwargs={},
            save_name=None):
        '''
        Compute the structure function.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        verbose : bool, optional
            Plot structure function transform.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        boundary : {"wrap", "fill"}, optional
            Use "wrap" for periodic boundaries, and "cut" for non-periodic.
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial break point guess. Enables fitting a segmented
            linear model.
        fit_kwargs : dict, optional
            Passed to `~turbustat.statistics.lm_seg.Lm_Seg.fit_model` when
            using a broken linear fit.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.compute_structfunc2d(boundary=boundary,
                                  show_progress=show_progress)
        self.fit_plaw(xlow=xlow, xhigh=xhigh, brk=brk, verbose=verbose,
                      **fit_kwargs)

        if verbose:
            self.plot_fit(save_name=save_name, xunit=xunit)

        return self
