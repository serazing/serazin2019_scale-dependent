import numpy as np
import xarray as xr
import scipy

# energy parameter
E = 6.3e-5

# j_*
js = 3

# sum_1^infty (j^2+j_*^2)^-1
jsum = ((np.pi * js / np.tanh(np.pi * js) - 1) 
        / (2 * js ** 2))

# gravitational acceleration
g = 9.81

def generate_kh(min_decade=-6, max_decade=-2, nb_points=401):
    kh = xr.IndexVariable('kh', 2 * np.pi * np.logspace(min_decade,
                                                        max_decade, 
                                                        nb_points),
                          attrs={'long_name': 'Horizontal wavelength', 
                                 'units' :'rad.m-1'})
    return kh


def generate_k(min_decade=-6, max_decade=-2, nb_points=401):
    k = xr.IndexVariable('k', 2 * np.pi * np.logspace(min_decade,
                                                      max_decade, 
                                                      nb_points),
                         attrs={'long_name': 'Wavelength', 
                                'units' :'rad.m-1'})
    return k


def generate_omega(N, f, nb_points=401):
    omega = xr.IndexVariable('omega', np.logspace(np.log10(1.01 * f),
                                                  np.log10(N), 
                                                  nb_points),
                             attrs={'name': 'Frequency', 
                                    'units' :'rad.s-1'})
    return omega


def generate_modes(nb_modes=100):
    j = xr.IndexVariable('mode', np.arange(1, nb_modes),
                         attrs={'name': 'Vertical mode'})
    return j



def fit_exponential_profile(N, dim=''):
    slope, intercept, _, _, _ = scipy.stat.linregress(N[dim], np.log(N))
    b = 1. / slope
    N0 = np.exp(intercept)

    
def omega_from_k_j(k, j, f, N0, e):
    """ 
    Compute frequency omega as a function of horizontal wavenumber k 
    and mode number j
    
    Parameters
    ----------
    k : array_like
        Horizontal wavenumbers in m-1
    j : array_like
        Vertical modes numbers
    f : float
        Coriolis frequency
    N0 : float
        Surface buoyancy frequency
    e : float
        E-folding depth of N(z)
    """
    num = N0 ** 2 * k ** 2 + f ** 2 * (np.pi * j / e) ** 2
    den = k ** 2 + (np.pi * j / e) ** 2
    omega = np.sqrt(num / den)
    return omega


def k_omega_j(omega, j, f, N0, b):
    # hor. wavenumber as a function of frequency omg and mode number j
    num = (omega ** 2 - f ** 2)
    den = (N0 ** 2 - omega ** 2)
    k = np.pi * j / b * np.sqrt(num / den)
    return k


def B(omg, f):
    # Munk's B(omg) describing the frequency distribution
    return 2/np.pi*f/omg/np.sqrt(omg**2-f**2)


def H(j):
    # Munk's H(j) describing the mode distribution
    return 1./(j**2+js**2)/jsum


def E_omg_j(omg, j, f):
    # Munk's E(omg,j)
    return B(omg, f)*H(j)*E


def E_k_j(k, j, f, N, N0, b):
    # Munk's E(omg,j) transformed into hor. wavenumber space:
    # E(k,j) = E(omg,j) domg/dk. The transformation is achieved using the
    # dispersion relation (9.23a) in Munk (1981).
    omg = omega_from_k_j(k, j, f, N0, b)
    domgdk = ((N0 ** 2 - omg ** 2) 
              / omg * k 
              / (k ** 2 + (np.pi * j / b) ** 2))
    return E_omg_j(omg, j, f)*domgdk


def pe_k_j(k, j, f, N, N0, b):
    """
    Potential energy spectrum (N^2 times displacement spectrum) as a
    function of hor. wavenumber k and mode number j
    """
    omega = omega_from_k_j(k, j, f, N0, b)
    num = (b ** 2 * N0 * N * (omega ** 2 - f ** 2)) 
    den = omega ** 2
    pe = num / den * E_k_j(k, j, f, N, N0, b)
    return xr.DataArray(pe, coords=(k, j), name='PE')


def ke_k_j(k, j, f, N, N0, b):
    """
    Kinetic energy spectrum as a function of hor. wavenumber k and mode
    number j
    """
    omega = omega_from_k_j(k, j, f, N0, b)
    num = (b ** 2 * N0 * N * (omega ** 2 + f ** 2))
    den = omega ** 2
    ke =  num / den * E_k_j(k, j, f, N, N0, b)
    return xr.DataArray(ke, coords=(k, j), name='KE')


def eta_k_j(k, j, f, N, N0, b):
    """
    SSH spectrum as a function of hor. wavenumber k and mode number j
    """ 
    omega = omega_from_k_j(k, j, f, N0, b)
    num_a = (omega ** 2 - f ** 2) ** 2
    den_a = (f ** 2 * (omega ** 2 + f ** 2))
    num_b = ke_k_j(k, j, f, N, N0, b)
    den_b = k ** 2 
    num_c = f ** 2
    den_c = g ** 2
    eta = num_a / den_a * num_b / den_b * num_c /den_c
    return xr.DataArray(eta, coords=(k, j), name='eta')


def pe_omega_j(omega, j, f, N, N0, b):
    # potential energy spectrum (N^2 times displacement spectrum) as a function
    # of frequency omg and mode number j
    num = (b ** 2 * N0 * N * (omega ** 2 - f ** 2)) 
    den = omega ** 2
    pe = num / den * E_omg_j(omega, j, f)
    return xr.DataArray(pe, coords=(omega, j), name='PE') 


def ke_omega_j(omega, j, f, N, N0, b):
    # kinetic energy spectrum as a function of frequency omg and mode number j
    num = (b ** 2 * N0 * N * (omega ** 2 + f ** 2)) 
    den = omega ** 2
    ke = num / den * E_omg_j(omega, j, f)
    return xr.DataArray(ke, coords=(omega, j), name='KE') 


def eta_omega_j(omega, j, f, N, N0, b):
    # SSH spectrum as a function of frequency omg and mode number j
    k = k_omega_j(omega, j, f, N0, b)
    
    num_a = (omega ** 2 - f ** 2) ** 2
    den_a = (f ** 2 * (omega ** 2 + f ** 2))
    num_b = ke_omega_j(omega, j, f, N, N0, b)
    den_b = k ** 2 
    num_c = f ** 2
    den_c = g ** 2
    eta = num_a / den_a * num_b / den_b * num_c /den_c
    return xr.DataArray(eta, coords=(omega, j), name='eta') 


def duu_r(r, f=7.3e-5, N=2.4e-3, N0=5.2e-3, b=1.3e3):
    """
    Compute the second order structure function of horizontal velocities 
    correspondin to the Garrett and Munk spectrum. N0 and the e-folding
    scale are determined from an exponential vertical profile.
    
    Parameters
    ----------
    r : array_like
        Seperation vectore
    N : float, optional
        Buoyancy frequency
    N0 : float, optional
        Surface-extrapolated buoyancy frequency
    b : float, optional
        E-folding scale of N(z)
        
    Returns
    -------
    duu : xr.DataArray
        The second order structure function of horizontal velocities
    """
    # Generate horizontal wavenumbers and modes 
    kh = generate_kh()
    j = generate_modes()
    # Compute the wavenumber spectrum 
    ke_kh_j = ke_k_j(kh, j, f, N, N0, b)
    ke_kh = ke_kh_j.sum('mode')
    # Convert the spectrum to a structure function
    duu = spectrum_to_structure_function(ke_kh, r, dim='kh')
    duu = duu.assign_coords(r=r)
    return duu

def drhorho_r(r, f=7.3e-5, N=2.4e-3, N0=5.2e-3, b=1.3e3):
    """
    Compute the second order structure function of density
    corresponding to the Garrett and Munk spectrum. N0 and the e-folding
    scale are determined from an exponential vertical profile.
    
    Parameters
    ----------
    r : array_like
        Seperation vectore
    N : float, optional
        Buoyancy frequency
    N0 : float, optional
        Surface-extrapolated buoyancy frequency
    b : float, optional
        E-folding scale of N(z)
        
    Returns
    -------
    duu : xr.DataArray
        The second order structure function of horizontal velocities
    """
    rho_0 = 1025
    g = 9.81
    # Generate horizontal wavenumbers and modes 
    kh = generate_kh()
    j = generate_modes()
    # Compute the wavenumber spectrum 
    pe_kh_j = pe_k_j(kh, j, f, N, N0, b)
    pe_kh = pe_kh_j.sum('mode')
    # Convert the spectrum to a structure function
    drhorho = N ** 2 * rho_0 ** 2 / g ** 2 * spectrum_to_structure_function(pe_kh, r, dim='kh')
    drhorho = drhorho.assign_coords(r=r)
    return drhorho



def compute_1d_spectrum(k, spectrum_kh):
    kh = xr.DataArray(spectrum_kh['kh'])
    dkh = kh.diff('kh') # Differentiate the wavenumber vector
    spectrum_k = xr.DataArray(np.zeros(len(k)), coords=(k,))
    for i in range(len(k)):
        k_int = k.isel(k=i)
        spectrum_int = spectrum_kh.where(kh > k_int) / np.sqrt(kh ** 2 - k_int ** 2)
        spectrum_k[i] = 2 / np.pi * (spectrum_int  * dkh).sum('kh')
    return spectrum_k


def spectrum_to_structure_function(spectrum, r, dim='k'):
    from scipy.special import j0
    k = spectrum[dim]
    dk = k.diff(dim) # Differentiate the wavenumber vector
    Duu = 2 * (spectrum * (1 - j0(k * r)) * dk).sum(dim)
    return Duu 


