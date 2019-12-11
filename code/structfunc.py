import dask.delayed as delayed
import xarray as xr
import numpy as np

EARTH_RADIUS = 6371 * 1e3

@delayed(pure=True)
def lagged_difference(a, dim, lag, order=1):
    return (a.shift(**{dim: -lag}) - a) ** order

@delayed(pure=True)
def lagged_sum(a, dim, lag, order=1):
    return (a.shift(**{dim: -lag}) + a) ** order

@delayed(pure=True)
def norm(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)

@delayed(pure=True)
def longitudinal_structure_function(du, dv, dx, dy, order=2):
    """Scalar product between velocity differences and separation vector"""
    dnl = ((du * dx + dv * dy) / np.sqrt(dx ** 2 + dy ** 2)) ** order
    return dnl

@delayed(pure=True)
def transverse_structure_function(du, dv, dx, dy, order=2):
    """Cross product between velocity differences and separation vector"""
    dnt = ((dv * dx - du * dy) / np.sqrt(dx ** 2 + dy ** 2)) ** order
    return dnt

@delayed(pure=True)
def dlat2dy(dlat):
    dy = np.pi / 180. * EARTH_RADIUS * dlat
    return dy

@delayed(pure=True)
def dlon2dx(dlon, lat, dim):
    dx = (np.cos(np.pi / 180. * lat) * 
          np.pi / 180. * EARTH_RADIUS * dlon)
    return dx


def velocity_structure_function(data, dim, max_lag, error=None, order=2):
    output = xr.Dataset()
    output_vars = {}
    dr_list, dnl_list, dnt_list, err_list = [], [], [], []
    for lag in range(1, max_lag):
        # Compute velocity differences
        du = lagged_difference(data['u'], dim, lag)
        dv = lagged_difference(data['v'], dim, lag)
        # Compute latitudinal and longitudinal differences
        dlat = lagged_difference(data['lat'], dim, lag)
        dlon = lagged_difference(data['lon'], dim, lag)
        # Compute spherical distance between points
        dx = dlon2dx(dlon, data['lat'], dim)
        dy = dlat2dy(dlat)
        dr = norm(dx, dy) * 1e-3
        # Compute longitudinal and transverse velocity structure functions
        dnl = longitudinal_structure_function(du, dv, dx, dy, order=order)
        dnt = transverse_structure_function(du, dv, dx, dy, order=order)
        # Append results into lists
        dr_list.append(dr)        
        dnl_list.append(dnl)
        dnt_list.append(dnt)
        # Try to get some info about errors
        if error is not None:
            err = lagged_sum(np.abs(data[error]), dim, lag, order=order)
            err_list.append(err)
    # Concatenate lists and store the results into a Dataset        
    Dnl = delayed(xr.concat)(dnl_list, dim='lags').stack(r=(dim, 'lags'))    
    Dnt = delayed(xr.concat)(dnt_list, dim='lags').stack(r=(dim, 'lags'))
    r = delayed(xr.concat)(dr_list, dim='lags').stack(r=(dim, 'lags'))    
    output_vars['D%sl' % order] = Dnl.compute()
    output_vars['D%st' % order] = Dnt.compute()
    if error is not None:
        D_err = delayed(xr.concat)(err_list, dim='lags').stack(r=(dim, 'lags'))
        output_vars['D%s_err' % order] = D_err.compute()    
    output = output.assign(**output_vars)
    output['r'] = r.compute()
    return output


def tracer_structure_function(data, dim, max_lag, error=None, order=2):
    output = xr.Dataset()
    output_vars = {}
    dr_list, dSST_list, dSSS_list, dSSb_list = [], [], [], []
    for lag in range(1, max_lag):
        # Compute velocity differences
        dSST = lagged_difference(data['SST'], dim, lag, order=2) 
        dSSS = lagged_difference(data['SSS'], dim, lag, order=2)
        dSSb = lagged_difference(data['SSrho'], dim, lag, order=2)
        # Compute latitudinal and longitudinal differences
        dlat = lagged_difference(data['lat'], dim, lag)
        dlon = lagged_difference(data['lon'], dim, lag)
        # Compute spherical distance between points
        dx = dlon2dx(dlon, data['lat'], dim)
        dy = dlat2dy(dlat)
        dr = norm(dx, dy) * 1e-3
        # Append results into lists
        dr_list.append(dr)        
        dSST_list.append(dSST)
        dSSS_list.append(dSSS)
        dSSb_list.append(dSSb)
    # Concatenate lists and store the results into a Dataset        
    DnSST = delayed(xr.concat)(dSST_list, dim='lags').stack(r=(dim, 'lags'))
    DnSSS = delayed(xr.concat)(dSSS_list, dim='lags').stack(r=(dim, 'lags'))
    DnSSb = delayed(xr.concat)(dSSb_list, dim='lags').stack(r=(dim, 'lags'))
    r = delayed(xr.concat)(dr_list, dim='lags').stack(r=(dim, 'lags'))    
    output_vars['D%sSST' % order] = DnSST.compute()
    output_vars['D%sSSS' % order] = DnSSS.compute()
    output_vars['D%sSSrho' % order] = DnSSb.compute()
    output = output.assign(**output_vars)
    output['r'] = r.compute()
    return output


def helmholtz_decomposition(d2, dim, ul='l', ut='t'):
	"""
	Perform a Helmholtz decomposition of the second order structure functions

	Parameters
	----------
	d2 : xarray.DataSet
		The Dataset containing the structure functions
	dim : str
		The length dimension
	ul : str, optional
		The longitudinal velocity name. Default is 'u'.
	ut : str, optional
		The transverse velocity name, default is 'v'

	Returns
	-------
	res  :  xarray.DataSet
	The structure functions averaged
	"""
	d2l = 'D2%s' % ul
	d2t = 'D2%s' % ut
	r = d2[dim].squeeze()
	from scipy.integrate import cumtrapz
	integral = cumtrapz((1. / r * (d2[d2t] - d2[d2l])).data,
	                     x=r, initial=0, axis=d2[d2t].get_axis_num(dim))
	d2r = d2[d2t] + integral
	d2d = d2[d2l] - integral
	return d2.assign(D2r=d2r, D2d=d2d)


def average_structure_function(dn, r_bins, mode='mean'):
	"""
	Average the structure functions over different bins using
	:py:func:`xarray.Dataset.groupby_bins`

	Parameters
	----------
	dn : xarray.DataSet
		The Dataset containing the nth structure functions
	r_bins :  int or array of scalars
		The list of bins over which the structure functions are averaged (see
		:py:func:`xarray.Dataset.groupby_bins`)
	mode : {'mean', 'median'}, optional
		Define if the averaged is performed by using the mean (default) or
		the median

	Returns
	-------
	res : xarray.DataSet
		The structure functions averaged
	"""
	r_labels = r_bins[1:] - np.diff(r_bins) / 2
	group = dn.groupby_bins('r', r_bins, labels=r_labels)
	if mode is 'mean':
		avg_dn = group.mean(dim='r')
	elif mode is 'median':
		avg_dn = group.median(dim='r')
	else:
		raise(ValueError, "This mode is not available.")
	nobs = group.count(dim='r')
	nobs_vars = {'nobs_' + var: nobs[var] for var in nobs.data_vars}
	return avg_dn.assign(**nobs_vars)