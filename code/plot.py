import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmclimate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numba

import gm
import structfunc
import quality as qc

def add_map(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
            central_longitude=0., scale='auto', ax=None):
    """
    Add the map to the existing plot using cartopy

    Parameters
    ----------
    lon_min : float, optional
        Western boundary, default is -180
    lon_max : float, optional
        Eastern boundary, default is 180
    lat_min : float, optional
        Southern boundary, default is -90
    lat_max : float, optional
        Northern boundary, default is 90
    central_longitude : float, optional
        Central longitude, default is 180
    scale : {‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, ‘full’}, optional
        The map scale, default is 'auto'
    ax : GeoAxes, optional
        A new GeoAxes will be created if None

    Returns
    -------
    ax : GeoAxes
    Return the current GeoAxes instance
    """
    import cartopy.feature as cfeature
    extent = (lon_min, lon_max, lat_min, lat_max)
    if ax is None:
        ax = plt.subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(central_longitude=central_longitude))
    ax.set_extent(extent)
    #land = cfeature.GSHHSFeature(scale=scale,
    #                             levels=[1],
    #                             facecolor=cfeature.COLORS['land'])
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='black',
                      alpha=0.5, xlocs=range(lon_min - 1, lon_max + 1, 5), 
                      ylocs=range(lat_min - 1, lat_max + 1, 5))
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def plot_regions(poly, lat_min=-40, lat_max=-5, lon_min=150, lon_max=180, scale='low', 
                 lw=2, edgecolor='red', facecolor='white', alpha=0.5, ax=None):
    """
    Plot ADCP transects from regions of intres
    """
    ax = add_map(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, 
                 lon_max=lon_max, scale=scale, ax=ax)
    if poly is not None:
        ax.add_geometries(poly, ccrs.PlateCarree(), edgecolor=edgecolor, 
                          facecolor=facecolor, alpha=alpha, lw=lw, zorder=10)
    plt.legend()
    plt.tight_layout()
    

def plot_transect(list_of_files, lon_name='lon', lat_name='lat', 
                  markersize=0.5, lat_min=-40, lat_max=-5, 
                  lon_min=150, lon_max=180):
    for filename in list_of_files:        
        try:
            ds = xr.open_dataset(filename).load()
            cond = ((ds[lon_name] >= lon_min) &
                    (ds[lon_name] <= lon_max) &
                    (ds[lat_name] >= lat_min) &
                    (ds[lat_name] <= lat_max)
                   )
            ds = ds.where(cond, drop=True)
        except ValueError:
            print('Check the data in %s' %filename)
        plt.plot(ds[lon_name], ds[lat_name], '.', 
                 markersize=markersize, color='black')


def plot_M2(M2_amp, lat_min=-40, lat_max=-5, lon_min=150, lon_max=180, vmax=4):
    M2_subset = M2_amp.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    M2_subset.plot.contourf(x='lon', y='lat', cmap=cmclimate.cm.BlGrYeOrReVi200, 
                            levels=25, vmax=vmax)        
    cbar = plt.gcf().get_axes()[1]
    cbar.set_ylabel('M2 amplitude $(cm)$') 

    
def plot_EKE(EKE,  lat_min=-40, lat_max=-5, lon_min=150, lon_max=180, **kwargs):
    EKE_subset = EKE.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    EKE_subset.plot.contourf(x='lon', y='lat', cmap=cmclimate.cm.WhiteBlueGreenYellowRed, 
                             **kwargs)
    cbar = plt.gcf().get_axes()[1]
    cbar.set_ylabel('Eddy Kinetic Energy ($cm^2.s^-2)$') 
    
    
def plot_segments(segments=None, mission_name='',  poly=None, 
                  lat_min=-22.5, lat_max=-20, 
                  lon_min=165.5, lon_max=168.5):
    """
    Plot ADCP transects from regions of intres
    """
    ax = add_map(lat_min=lat_min, lat_max=lat_max, 
                 lon_min=lon_min, lon_max=lon_max)
    for ds in segments:
        ax.plot(ds['lon'], ds['lat'], '.', markersize=2,
                label='%s_s%s' % (ds.attrs['mission'], 
                                  ds.attrs['segment_number']))
    land = cfeature.GSHHSFeature(scale='intermediate', levels=[1], 
                                 facecolor=cfeature.COLORS['land'])
    if poly is not None:
        ax.add_geometries(poly, ccrs.PlateCarree(), alpha=0.2)
    plt.title(mission_name)
    #plt.legend()
    plt.tight_layout()
    
    
#@numba.jit    
def bootstrap(D2, n=10000, ci=0.05):
    D2_mean = D2.median('segment')
    nseg = D2.sizes['segment']
    list_of_delta = []
    for i in range(n):
        D2_mean_boot = D2.isel(segment=np.random.randint(0, high=(nseg - 1), size=nseg)).median('segment')
        delta = D2_mean_boot - D2_mean
        list_of_delta.append(delta)
    bootstrap = xr.concat(list_of_delta, dim='nboot')
    lower_ci = D2_mean + bootstrap.quantile(ci, dim='nboot')
    upper_ci = D2_mean + bootstrap.quantile(1 - ci, dim='nboot')
    return lower_ci, upper_ci

    
def plot_with_bootstrap(D2, label, color, lw=4, linestyle='-', **kwargs):
    D2_min, D2_max = bootstrap(D2)
    D2_mean = D2.median('segment')
    plt.loglog(D2_mean['r_bins'], D2_mean, 
               label=label, lw=lw, color=color, linestyle=linestyle)
    plt.fill_between(D2_mean['r_bins'], D2_min, D2_mean, color=color, **kwargs)
    plt.fill_between(D2_mean['r_bins'], D2_mean, D2_max, color=color, **kwargs)
    #plt.loglog(D2_mean['r_bins'], D2_min, 
    #           label=label, color=color, lw=2, ls='--')
    #plt.loglog(D2_mean['r_bins'], D2_max, 
    #           label=label, color=color, lw=2, ls='--')
    
    
def plot_helmholtz_decomposition(D2, segments=True,
                                 power_laws=True,
                                 ylim=[1e-3, 1], xlim=[1, 100]):
    #if segments:
    #    plot_total_structure_functions(D2, xlim=xlim, ylim=ylim)
    # Total structure functions
    D2_total = D2['D2l'] + D2['D2t']
    # Helmholtz decomposition
    list_of_D2 = []
    for seg in D2.segment:
        list_of_D2.append(structfunc.helmholtz_decomposition(D2.sel(segment=seg), 'r_bins'))
    D2_helmholtz = xr.concat(list_of_D2, dim='segment')
    #D2_mean = structfunc.helmholtz_decomposition(D2_mean, 'r_bins')
    # Longitudinal structure function
    #plt.loglog(D2_avg['r_bins'], D2_avg['D2l'] , 
    #           label="Longitudinal", lw=4, ls='--')
    # Transverse structure function
    #plt.loglog(D2_avg['r_bins'], D2_avg['D2t'] , 
    #           label="Transverse", lw=4, ls='--')
    # Total structure function
    plot_with_bootstrap(D2_total, label="Total", color='C2', alpha=0.5)
    # Rotational structure function
    plot_with_bootstrap(D2_helmholtz['D2r'], label="Rotational", color='C3', alpha=0.5)
    # Divergent structure function
    plot_with_bootstrap(D2_helmholtz['D2d'], label="Divergent", color='C4', alpha=0.5)
    ratio = (D2_helmholtz['D2d'].median('segment') / 
             D2_helmholtz['D2r'].median('segment')).mean()
    #ratio = (D2_helmholtz['D2d'] / 
    #         D2_helmholtz['D2r']).mean()
    # Ratio of Divergent and Rotational
    ax = plt.gca()
    text = r"$R_{\phi\psi} = %0.2f$" % ratio
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax.text(0.98, 0.03, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Plot power laws
    if power_laws:
        plot_power_laws(D2_total.median('segment'))
    # Plot parameters
    plt.xlabel(r'$r\,(km)$')
    plt.ylabel(r'$ D_{UU} \, (m^2.s^{-2})$')
    plt.xlim(xlim)
    plt.ylim(ylim)    
    plt.grid(which='both')

    
def plot_Rossby_radius(D2, lat=35, ylim=[1e-3, 1], xlim=[1, 100], **kwargs):
    D2_avg = D2.mean('segment')
    D2_avg = structfunc.helmholtz_decomposition(D2_avg, 'r_bins')
    # Latitude to Coriolis frequency
    omega_earth = 7.2921e-5
    f = 2 * omega_earth * np.sin(np.pi / 180. * lat) 
    Ro =  (np.sqrt(D2_avg['D2l'] + D2_avg['D2t']) / 
           (1e3 * D2_avg['r_bins'] * abs(f)))
    R = D2_avg['D2d'] / D2_avg['D2r']
    # Transverse structure function
    plt.loglog(D2_avg['r_bins'], Ro, lw=4, **kwargs)
    # Plot parameters
    plt.xlabel(r'$r\,(km)$')
    plt.ylabel(r'$ R_o(r)$')
    plt.xlim(xlim)
    plt.ylim(ylim)    
    plt.grid(which='both')
    
    
def plot_divergent_rotational_ratio(D2, xlim=[1, 100], **kwargs):
    D2_avg = D2.mean('segment')
    D2_depth = D2_avg.groupby('depth').apply(structfunc.helmholtz_decomposition, args=('r_bins',)).mean('r_bins')
    print(D2_depth)
    #D2_avg = structfunc.helmholtz_decomposition(D2_avg, 'r_bins')
    R = D2_depth['D2d'] / D2_depth['D2r']
    # Transverse structure function
    plt.plot(abs(R), R['depth'], lw=4, **kwargs)
    # Plot parameters
    #plt.xlabel(r'$r\,(km)$')
    #plt.ylabel(r'$ R$')
    #plt.xlim(xlim)  
    plt.grid(which='both')

    
def plot_gm_structure_function(lat=35, N=2.4e-3, N0=5.2e-3, b=1.3e3):
    ax = plt.gca()
    r = xr.IndexVariable('r', np.logspace(2, 6, 401))
    # Latitude to Coriolis frequency
    omega_earth = 7.2921e-5
    f = 2 * omega_earth * np.sin(np.pi / 180. * lat)    
    D2_GM = gm.duu_r(r, f=abs(f), N=N, N0=N0, b=abs(b))
    ax.loglog(1e-3 * r, D2_GM, lw=2, color='black', label='GM81')
    #plt.loglog(1e-3 * r, D2_GM, lw=3, color='black', label='GM81')

    
def plot_power_laws(D2):
    from xscale.spectral.tools import fit_power_law, plot_power_law 
    # Plot classic power laws
    power, scale_factor = fit_power_law(D2['r_bins'], D2)
    ax=plt.gca()
    plot_power_law(2., scale_factor, ls='--', lw=2, 
                   color='black', ax=ax, label=r'$r^2$')
    plot_power_law(1., scale_factor, ls='-.', lw=2, 
                   color='black', ax=ax, label=r'$r$')
    plot_power_law(2. / 3., scale_factor, ls=':', lw=2, 
                   color='black', ax=ax, label=r'$r^{2/3}$')
    text = "Estimated slope: %0.2f" % power
    # Properties of the box
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    # Place a text box in lower left in axes coords
    ax.text(0.71, 0.97, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    
def plot_total_structure_functions(D2, ylim=[1e-3, 1], xlim=[1, 100]):    
    for i in range(D2.sizes['segment']):
        D2_seg = D2.isel(segment=i)
        plt.loglog(D2['r_bins'], D2_seg['D2l'] + D2_seg['D2t'], 
                   lw=0.2, alpha=0.5, color='black', label='')
    # Plot parameters
    plt.xlabel(r'$r\,(km)$')
    plt.ylabel(r'$ D_{UU} \, (m^2.s^{-2})$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    

def plot_uship(raw_data, segments, mission_name='', ax=None):
    if ax is None:
        ax = plt.gca()
    ds_surf = raw_data.isel(depth=0)
    uship = xr.ufuncs.sqrt(ds_surf.uship ** 2 + ds_surf.vship ** 2)
    uship.plot(label='Raw data', ls='--', ax=ax)
    segment_nb = 1
    for ds in segments:
        ds_surf = ds.isel(depth=0)
        uship = xr.ufuncs.sqrt(ds_surf.uship ** 2 + ds_surf.vship ** 2)
        uship.plot(ax=ax)
        segment_nb +=1
    ax.set_ylabel(r'Ship velocity ($m.s^{-1}$)')
    ax.set_title(mission_name)    
    
    
def plot_heading(raw_data, segments, mission_name='', ax=None):
    if ax is None:
        ax = plt.gca()
    raw_data.isel(depth=0).heading.plot(label='Raw data', ls='--', ax=ax)
    segment_nb = 1
    for ds in segments:
        ds_surf = ds.isel(depth=0)
        ds_surf.heading.plot(ax=ax)
        segment_nb +=1 
    ax.set_ylabel(r'Ship heading ($^{\circ}$)')
    ax.set_title(mission_name)    
    
    
def monitor_segments(raw_data, segments, output_path, mission_name):
    full_output_path = "%s/%s/" % (output_path, mission_name)
    plt.figure()
    ax1 = plt.subplot(211)
    plot_uship(raw_data, segments, mission_name=mission_name, ax=ax1)
    ax2 = plt.subplot(212)
    plot_heading(raw_data, segments, mission_name=mission_name, ax=ax2)
    plt.tight_layout()
    plt.savefig(full_output_path + mission_name + '_ship_velocity_and_heading.png', dpi=300)
    plt.close('all')

    
def plot_segments(raw_data=None, segments=None, name='', output_path=None, 
                  poly=None, **kwargs):
    """
    Plot ADCP transects from regions of intres
    """
    plt.figure()
    ax = add_map(**kwargs)
    if raw_data is not None:
        ax.plot(raw_data.lon, raw_data.lat, '--', label='Raw data', lw=1)
    section_nb = 1
    if segments is not None:
	    for ds in segments:
	        ax.plot(ds.lon, ds.lat, '.',
	                label='Segment %s' % section_nb, markersize=2)
	        section_nb +=1
    if poly is not None:
        ax.add_geometries(poly, ccrs.PlateCarree(), alpha=0.2)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        full_output_path = "%s/%s/" % (output_path, name)
        plt.savefig(full_output_path + name + '_segment_map.png', 
                    dpi=300, bbox_inches='tight')
        plt.close('all')