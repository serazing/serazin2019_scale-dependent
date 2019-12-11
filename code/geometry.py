import xarray as xr
import numpy as np
import pandas as pd

EARTH_RADIUS = 6371 * 1e3


def latlon2yx(lat, lon):
    """
    Convert latitude and longitude arrays to y and x arrays in m

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates

    Returns
    -------
    y : array-like
        Zonal cartesian coordinates
    x : array-like
        Meridional cartesian coordinates
    """
    y = np.pi / 180. * EARTH_RADIUS * lat
    x = np.cos(np.pi / 180. * lat) * np.pi / 180. * EARTH_RADIUS * lon
    return y, x


def latlon2dydx(lat, lon, dim, label='upper'):
    """
    Convert latitude and longitude arrays to elementary displacements in dy
    and dx

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates
    dim : str
        Dimension along which the differentiation is performed, generally
        associated with the time dimension.
    label : {'upper', 'lower'}, optional
        The new coordinate in dimension dim will have the values of
        either the minuend’s or subtrahend’s coordinate for values ‘upper’
        and ‘lower’, respectively.

    Returns
    -------
    dy : array-like
        Zonal elementary displacement in cartesian coordinates
    dx : array-like
        Meridional elementary displacement in cartesian coordinates
    """
    dlat = lat.diff(dim, label=label)
    dlon = lon.diff(dim, label=label)
    dy = np.pi / 180. * EARTH_RADIUS * dlat
    # Need to slice the latitude data when there are duplicate values
    if label is 'upper':
        dx = (np.cos(np.pi / 180. * lat.isel(**{dim: slice(1, None)})) *
              np.pi / 180. * EARTH_RADIUS * dlon)
    elif label is 'lower':
        dx = (np.cos(np.pi / 180. * lat.isel(**{dim: slice(None, -1)})) *
              np.pi / 180. * EARTH_RADIUS * dlon)
    return dy, dx

def latlon2heading(lat, lon, dim, label='upper'):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    Parameters
    ----------

    Returns
    -------
      The bearing in degrees

    """
    dy, dx = latlon2dydx(lat, lon, dim, label=label)
    initial_heading = np.arctan2(dx, dy) * 180. / np.pi
    # Normalize the initial heading
    compass_heading = (initial_heading + 360) % 360
    return compass_heading


def latlon2vu(lat, lon, dim, label='upper'):
	"""
	Estimate the  meriodional and zonal velocity based on the
	latitude and longitude coordinates.

	Paramaters
	----------
    lat : xarray.DataArray
        Latitudinal spherical coordinates
    lon : xarray.DataArray
        Longitudinal spherical coordinates
    dim : str
        Name of the time dimension.

	Returns
	------
	v : xarray.DataArray
		The meridional velocity
	u : xarray.DataArray
		The zonal velocity
	"""
    # Compute spatial difference
	dy, dx = latlon2dydx(lat, lon, dim=dim, label=label)
    # Compute time difference in seconds
	dt = pd.to_numeric(lat[dim].diff(dim=dim, label=label)) * 1e-9
	v, u = dx / dt, dy / dt
	return v, u


def inpolygon(data, poly):
    """
    Mask the data outside a polygon using shapely. Data must have the
    longitudinal and latitudinal coordinates 'lon' and 'lat', respectively.

    Paramaters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to mask
    poly : BaseGeometry
        A polygon defined using shapely

    Returns
    -------
    res : xarray.DataArray or xarray.Dataset
        The data masked outside the poylgon
    """
    from shapely.geometry import Point
    lon = data['lon']
    lat =  data['lat']
    def inpolygon(polygon, xp, yp):
        return np.array([Point(x, y).intersects(polygon)
                         for x, y in zip(xp, yp)], dtype=np.bool)
    mask = inpolygon(poly, lon.data.ravel(), lat.data.ravel())
    da_mask = xr.DataArray(mask, dims=lon.dims, coords=lon.coords)
    return da_mask


def inpolygon2d(data, poly):
    from shapely.geometry import Point 
    lon = data['lon']
    lat =  data['lat']
    lon2d, lat2d = xr.broadcast(lon, lat)
    def inpolygon(polygon, xp, yp):
        return np.array([Point(x, y).intersects(polygon) for x, y in zip(xp, yp)], dtype=np.bool)
    mask = inpolygon(poly, lon2d.data.ravel(), lat2d.data.ravel()).reshape(lon2d.shape)
    da_mask = xr.DataArray(mask, dims=lon2d.dims, coords=lon2d.coords)
    return da_mask


def get_data_over_region(ds, region='NCAL', drop=False):
    from shapely.geometry import Polygon
    coords = dict()
    coords['SCAL'] = [[162, -23], [171, -23], [171, -27], [162, -27]]
    coords['ECAL'] = [[169, -20.5], [169, -24], [180, -24], [180, -20.5]]
    coords['NCAL'] = [[169, -20], [166, -20], [162, -18], [162, -12], 
                      [166, -12], [166, -15]]
    coords['VAUB'] = [[167.1, -22], [165.5, -21], [164.5, -20], [165.5, -20], 
                      [167, -21], [168.5, -22]]
    mask_region = inpolygon2d(ds, Polygon(coords[region]))
    ds_region = ds.where(mask_region, drop=drop)
    lon_mean, lat_mean =  np.mean(coords[region], axis=0)
    ds_region.attrs['lon_mean'] = lon_mean
    ds_region.attrs['lat_mean'] = lat_mean
    return ds_region