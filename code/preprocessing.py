import xarray as xr
import numpy as np
import geometry as geom 
import pandas as pd

def open_adcp_transect_from_olvac(filename):
    """
    Open an ADCP transect from the OLVAC dataset, and homogenize the coordinates
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``
    
    Returns
    -------
    ds : xarray.Dataset
        The ADCP transect under the form of a Dataset
    """
    vars_dropped =['dday', 'iblkprf', 'head_misalign', 'tr_temp', 'last_temp',
                   'swcor', 'amp', 'amp1', 'amp2', 'amp3', 'amp4',
                   'umean', 'vmean', 'umeas', 'vmeas', 'w', 'wmean',
                   'decday', 'woce_date', 'woce_time', 'TIME_bnds',
                   'trans_temp', 'sd_trans_temp', 'sd_ship_u', 'sd_ship_v', 'LAGON_FLAG', 
                   'BATHY', 'nanmask', 'TIME', 'sndv']
    ds = xr.open_dataset(filename, drop_variables=vars_dropped, autoclose=True)
    if 'NBINS' in ds.dims:
        ds = ds.rename({'NBINS' : 'nbins', 'NPROFS' : 'nprofs'})  
        time = ds.isel(nbins=0)['time']
        # Reshape the time counter
        df = pd.DataFrame({'year': time.isel(ntime=0), 'month': time.isel(ntime=1),
                           'day': time.isel(ntime=2), 'hour': time.isel(ntime=3),
                           'minute': time.isel(ntime=4), 'second': time.isel(ntime=5)})
        time_counter = pd.to_datetime(df)
        new_time = xr.DataArray.from_series(time_counter).rename({'index': 'time'})
        new_depth = ds.isel(nprofs=0)['depth'].rename({'nbins': 'depth'})
        ds = ds.drop(('time', 'depth')).rename({'nbins' : 'depth', 'nprofs' : 'time'}) 
        ds = ds.assign_coords(time=new_time, depth=new_depth)
    elif 'AXV' in ds.dims:
        ds = ds.rename({'AXV' : 'depth', 'TOTOT' : 'time',
                        'LONGITUDE': 'lon', 'LATITUDE': 'lat',
                        'U': 'u', 'V':'v'})
        # Reshape the time counter
        ref_time = ds['YEAR'].isel(time=0).astype('i4').item()
        time_counter = pd.to_datetime(ds['DECDAY'], unit='D', 
                                     origin=str(ref_time))
        ds['time'].data = time_counter
        ds = ds.drop(['DECDAY', 'YEAR', 'MONTH', 'HOURMS', 'AXV_bnds',
                      'BIN_PC_COV', 'LASTBIN'])
    elif 'AXISZ' in ds.dims:
        ds = ds.rename({'AXISZ' : 'depth', 'AXIST' : 'time',
                        'LONGITUDE': 'lon', 'LATITUDE': 'lat',
                        'U': 'u', 'V':'v'})
        # Reshape the time counter
        ref_time = ds['YEAR'][0].astype('i4').item()
        time_counter = pd.to_datetime(ds['DECDAY'], unit='D', 
                                     origin=str(ref_time))
        ds['time'].data = time_counter
        ds = ds.drop(['DECDAY', 'DAY', 'YEAR', 'MONTH', 'MASK',
                      'AXIST_bnds', 'AXISZ_bnds'])
    if 'longitude' in ds.variables:
        ds.rename({'longitude': 'lon', 'latitude': 'lat'}, inplace=True)
    if 'ship_u' in ds.variables:
        #ds = ds.drop('TIME').rename({'TIME':'time'})
        ds = ds.rename({'ship_u': 'uship', 'ship_v': 'vship'})
    if 'uship' not in ds.variables or 'vship' not in ds.variables:
        vship, uship = geom.latlon2vu(ds['lat'], ds['lon'], dim='time')
        ds = ds.assign(uship=uship, vship=vship)
    if 'heading' not in ds.variables:
        heading = geom.latlon2heading(ds['lat'], ds['lon'], dim='time')
        ds = ds.assign(heading=heading)
    else:
        if ds['heading'].std() < 5 or ds['heading'].std() > 360:
            heading = geom.latlon2heading(ds['lat'], ds['lon'], dim='time')
            ds = ds.assign(heading=heading)
    ds = clean_data(ds)
    # Sort by time to avoid strange behaviours
    ds = ds.sortby('time')
    return ds

def clean_data(ds, data_quality=80, max_velocity=100):
    """
    Clean unvalid values from the data
    """
    lon = (ds['lon'] + 360) % 360
    lat = ds['lat']
    uship = ds['uship']
    vship = ds['vship']
    ds['lon'].data = lon.data
    coordinates_check = (lon >= 0) & (lon <= 360) & (lat >= -90) & (lat <= 90)
    shipdata_check = (abs(ds['uship']) < max_velocity) & (abs(ds['vship']) < max_velocity)
    new_ds = ds.where(coordinates_check & shipdata_check, drop=True)   
    if 'pg' in new_ds.variables:
        quality_check = (new_ds['pg'] >= data_quality)
    else:
        quality_check = 1
    velocity_check = (abs(new_ds['u']) < max_velocity) & (abs(new_ds['v']) < max_velocity)
    new_u = new_ds['u'].where(velocity_check)
    new_v = new_ds['v'].where(velocity_check)
    return new_ds.assign(u=new_u, v=new_v)


def compute_ship_heading(lon, lat):
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
    deg2rad = np.pi / 180.
    lon1 = deg2rad * lon.shift(time=-1) 
    lon2 = deg2rad * lon  
    lat1 = deg2rad * lat.shift(time=-1)
    lat2 = deg2rad * lat 
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)* np.cos(lat2) * np.cos(lon2 - lon1))
    initial_heading = np.arctan2(x, y) * 180. / np.pi
    # Normalize the initial heading
    compass_heading = (initial_heading + 180) % 360 
    return compass_heading


def open_tsg_from_legos(filename):
    """
    Open thermosalinograph (TSG) transect from the LEGOS dataset, 
    and homogenize the coordinates
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    
    Returns
    -------
    ds : xarray.Dataset
        The TSG transect under the form of a Dataset
    """
    renamed_var = {'TIME': 'time', 'LON': 'lon', 'LAT': 'lat'}    
    ds = (xr.open_dataset(filename, autoclose=True)
            .rename(renamed_var)
            .set_coords(('lon', 'lat'))
         )
    ds['lon'] = (ds['lon'] + 360) % 360
    #Remove duplicated time values
    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    # Ship velocity
    if 'uship' not in ds.variables or 'vship' not in ds.variables:
        vship, uship = geom.latlon2vu(ds['lat'], ds['lon'], dim='time')
        ds = ds.assign(uship=uship, vship=vship)
    # Ship velocity
    if 'heading' not in ds.variables:
        heading = geom.latlon2heading(ds['lat'], ds['lon'], dim='time')
        ds = ds.assign(heading=heading)
    else:
        if ds['heading'].std() < 5 or ds['heading'].std() > 360:
            heading = geom.latlon2heading(ds['lat'], ds['lon'], dim='time')
            ds = ds.assign(heading=heading)
    # Sort by time to avoid strange behaviours
    ds = ds.sortby('time')
    for var in ds.variables:
        try:
            del(ds[var].attrs['coordinates'])
        except(KeyError):
            pass
    return ds


def open_argo_TS(T_file, S_file):
    ARGO_T = xr.open_dataset(T_file, decode_times=False, chunks={'TIME': 40})
    ARGO_S = xr.open_dataset(S_file, decode_times=False, chunks={'TIME': 40})
    Temperature = ARGO_T['ARGO_TEMPERATURE_MEAN'] + ARGO_T['ARGO_TEMPERATURE_ANOMALY']
    Salinity = ARGO_S['ARGO_SALINITY_MEAN'] + ARGO_S['ARGO_SALINITY_ANOMALY']
    ds = xr.Dataset({'Temperature': Temperature, 
                     'Salinity': Salinity})
    ds = ds.rename({'LONGITUDE': 'lon', 'LATITUDE': 'lat',
                    'PRESSURE': 'p', 'TIME': 'time'})
    ds['lon'] = (ds['lon'] + 360) % 360
    df = pd.DataFrame({'year': 2004 + ds['time'].data // 12 + 1, 
                   'month': (ds['time'].data - 0.5) % 12 + 1,
                   'day': 15 * np.ones_like(ds['time'].data)})
    time = pd.to_datetime(df, unit='D')
    ds['time'] = xr.DataArray.from_series(time).rename({'index': 'time'})
    return ds                                  


def open_argo_MLD_climatology(MLD_file):
    ds = xr.open_dataset(MLD_file)
    ds = ds.rename({'iLAT': 'lat', 'iLON': 'lon', 'iMONTH': 'month'})
    #sds['lon'] = (ds['lon'] + 360) % 360
    return ds                                  