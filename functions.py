import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp


def theta(t, A, tau, w, phase, h):
    return A * np.exp(-t/tau) * np.sin(w*t + phase) + h


def lin(x, a, b):
    return a*x + b

def r(a, o):
    b = ufloat(44, 0.01 / 3**0.5) * 1e-3
    c = ufloat(117.12, 0.01 / 3**0.5) * 1e-3
    d = ufloat(49.90, 0.03 / 3**0.5) * 1e-3
    e = ufloat(25.04, 0.01 / 3**0.5) * 1e-3
    p = (ufloat(120, 0.01 / 3**0.5) * 0.5 - ufloat(17.1, 0.01 / 3**0.5) * 0.5) * 1e-3
    l = ufloat(4.321, 0.001)
    
    beta = a - 0.5 * b - 0.5 * d
    gamma = c - 0.5 * d - 0.5 * e

    phi = np.arctan(o / l) + np.arcsin(beta / gamma)

    return (gamma**2 + p**2 - 2 * gamma * p *np.cos(phi))


def rotate_data(xdata, ydata):
    """
    Rotate 2D data so that its dominant linear trend aligns with the x-axis.

    This function first fits a straight line `y = m*x + b` to the input data
    using `curve_fit` and the model function `lin`. The fitted slope is then
    used to compute a rotation angle that removes the linear tilt of the data.
    The data are translated and rotated such that the fitted line becomes
    horizontal.

    Parameters
    ----------
    xdata : array-like
        Input x-coordinates.
    ydata : array-like
        Input y-coordinates.

    Returns
    -------
    x : ndarray
        Rotated x-coordinates, aligned with the principal direction of the data.
    y : ndarray
        Rotated y-coordinates, centered and orthogonal to the fitted trend.

    Notes
    -----
    - The rotation angle is computed as phi = arctan(m), where m is the fitted slope.
    - The data are shifted by the fitted intercept before rotation to center them.
    - Requires a linear model function `lin` compatible with `curve_fit`.
    """
    
    popt, pcov = curve_fit(lin, xdata=xdata, ydata=ydata)
    phi = np.arctan(popt[0])
    
    x = xdata * np.cos(-phi) - (ydata - popt[1]) * np.sin(-phi)
    y = xdata * np.sin(-phi) + (ydata - popt[1]) * np.cos(-phi)

    return x, y


def fit(t, xdata, ydata, p0: list = [180, 2000, 0.017, -1.5, 630], cutoff: int = 0,
        plot: bool = False, duration: bool = False, angle: bool = False):
    """
    Fit the expected model to rotated raw tracking data and optionally return
    derived quantities or a plot.

    The function first rotates the input data using `rotate_data`, then fits
    the model function `theta` to the rotated x-component using nonlinear
    least squares (`scipy.optimize.curve_fit`). From the fitted parameters,
    it computes the oscillation period T and optionally the angular offset.

    Parameters
    ----------
    t : array-like
        Time values corresponding to the data points.
    xdata : array-like
        Raw x-coordinate data.
    ydata : array-like
        Raw y-coordinate data.
    p0 : list, optional
        Initial guess for the fit parameters passed to `curve_fit`.
    cutoff : int, optional
        Index at which to start the fit (useful for ignoring initial data).
        Default is 0.
    plot : bool, optional
        If True, generate a two-panel plot showing:
        - The rotated x data with the fitted model.
        - The rotated y data.
        Default is False.
    duration : bool, optional
        If True, return the oscillation period T (with uncertainty).
        Default is False.
    angle : bool, optional
        If True, return the fitted angular offset parameter (with uncertainty).
        Default is False.

    Returns
    -------
    uncertainties.UFloat or numpy.ndarray or tuple
        - If `duration` and `angle` are both True:
            Returns (T, angle_offset), both as `ufloat`.
        - If only `duration` is True:
            Returns T as a `ufloat`.
        - If only `angle` is True:
            Returns the angular offset as a `ufloat`.
        - Otherwise:
            Returns all fitted parameters as an array with uncertainties
            (`uncertainties.unumpy.uarray`).

    Notes
    -----
    - The period is computed as T = 2π / ω, where ω is the third fitted parameter.
    - Parameter uncertainties are derived from the covariance matrix returned
      by `curve_fit`.

    """

    x, y = rotate_data(xdata, ydata)
    popt, pcov = curve_fit(theta, t[cutoff:], x[cutoff:], p0=p0)

    T = 2 * np.pi / ufloat(popt[2], pcov[2,2]**0.5)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(t, x - popt[4], 'k.', ms=1)
        ax1.plot(t[cutoff:], theta(t[cutoff:], *popt) - popt[4], 'r')
        #ax1.set_xlabel('t')
        ax1.set_ylabel('x')

        ax2.errorbar(t, y, fmt='k.', capsize=1, linewidth=0.6, ms=0.8)
        ax2.set_xlabel('t')
        ax2.set_ylabel('y')
        plt.show()

    if duration and angle:
        return T, ufloat(popt[4], pcov[4,4]**0.5)
    
    if duration:
        return T
    
    if angle:
        return ufloat(popt[4], pcov[4,4]**0.5)
    
    return unp.uarray(popt, np.sqrt(np.diag(pcov)))


# calculate G , incl uncertainties
def G(theta1: ufloat, theta2: ufloat, T, conversion_factor):
    #technical drawing
    m = ufloat(0.028, 0)  #0.028 #kg tech drawing
    l = ufloat(0.0429, 0.001) #uncert tech drawing
    I = m * l**2 / 2 #kg m^2, using MIT estimated formula
    
    # laser meas
    L = ufloat(4.321, 0.001) #m
        
    #measured:
    M = ufloat(1.5, 0.01)  #kg +/-10g
    
    r1 = ufloat(0.051722, 0.00011) #m  51.722+/-0.011
    r2 = ufloat(0.052152, 0.00011) #m  52.152+/-0.011
        
    #fit: T0, dtheta1, dtheta2
    h1 = theta1 * conversion_factor #conversion factor for vid setup 1 in cm/'m'. used squared addition here
    h2 = theta2 * conversion_factor
    
    dtheta= 2 * unp.arctan((0.5 * (h2 - h1)) * 0.01 / 4.321) #0.01 go from cm to m
    
    #formula static deflection
    delta = 0.002 #m
    c1 = m * M * l / r1**2
    c2 = -m * M * l / r2**2 
    
    G = 4 * np.pi**2 * I / T**2 * dtheta / (c1 - c2)  #units kg m^2 /s^2  * m^2/kg^2/m
    
    return G