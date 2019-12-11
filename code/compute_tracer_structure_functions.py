#!/usr/bin/env python

# Useful python module
import xarray as xr
import numpy as np
import pandas as pd
import gsw
import glob
import argparse

# Local modules
import structfunc as sf
import quality as qc


# Define the parser for input parameters

description = ("Compute structure functions on several segments")
parser = argparse.ArgumentParser(description=description)
parser.add_argument('inpath', help="Input path")
parser.add_argument('outfile', help="Output file")

args = parser.parse_args()


def open_segments(input_path):
    list_of_path = sorted(glob.glob(input_path + "/*/*_segment_*.nc"))
    return [xr.open_dataset(path, autoclose=True) for path in list_of_path]

def compute_density(data):
    p = 0 * data['lon']
    SA = gsw.SA_from_SP(data['SSS'], p, data['lon'], data['lat'])
    CT = gsw.CT_from_t(SA, data['SST'], p)
    rho, alpha, beta = gsw.rho_alpha_beta(SA, CT, p)
    #b = 9.81 * (1 - rho / 1025.)
    data['SSS'].data = SA
    data['SST'].data = CT
    return data.assign(SSrho=xr.DataArray(rho, dims='time'),
                       SSalpha=xr.DataArray(alpha, dims='time'),
                       SSbeta=xr.DataArray(beta, dims='time'))


segments = open_segments(args.inpath)
D2_list = []
for seg in segments:
    seg = seg.set_coords(('lon', 'lat', 'heading', 'uship', 'vship'))
    length = qc.get_segment_length(seg).mean()
    maxlength = length // 4
    nobs = seg.sizes['time']
    maxlag = nobs
    seg = compute_density(seg)
    # Computation and bining of structure functions
    D2 = sf.tracer_structure_function(seg.load(), dim='time', max_lag=maxlag)
    D2_valid = D2
    r_bins = 10 ** np.arange(0.3, np.log(maxlength), 0.1)
    D2_avg = sf.average_structure_function(D2_valid, r_bins, mode='mean')
    D2_avg = D2_avg.assign_coords(lat=seg['lat'].mean(),
                                  lon=seg['lon'].mean(),
                                  mission=seg.attrs['mission'],
                                  segment_number=seg.attrs['segment_number'],
                                  time=pd.to_datetime(np.mean(pd.to_numeric(seg['time'].data))),
                                  alpha=seg['SSalpha'].mean(),
                                  beta=seg['SSbeta'].mean()
                                 )    
    D2_list.append(D2_avg)
D2_all = xr.concat(D2_list, dim='segment')
D2_all.to_netcdf(args.outfile)