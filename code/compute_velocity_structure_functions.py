#!/usr/bin/env python

# Useful python module
import xarray as xr
import numpy as np
import pandas as pd
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


def open_ADCP_segments(input_path):
    list_of_path = sorted(glob.glob(input_path + "/*/*_segment_*.nc"))
    return [xr.open_dataset(path, autoclose=True) for path in list_of_path]


segments = open_ADCP_segments(args.inpath)
D2_list = []
for seg in segments:
    seg = seg.set_coords(('lon', 'lat', 'heading', 'uship', 'vship'))
    length = qc.get_segment_length(seg).mean()
    maxlength = length // 4
    # Get only data with pflag equals 0
    seg = seg.where(seg['pflag'] == 0)
    # Rotate velocity vectors to get longitudinal and tranverse velocities
    nobs = seg.sizes['time']
    maxlag = nobs
    # Computation and bining of structure functions
    D2 = sf.velocity_structure_function(seg, dim='time', max_lag=maxlag, error='e')
    #D2_valid = D2.where((D2['D2t'] > D2['D2_err']) & 
    #                    (D2['D2l'] > D2['D2_err']))
    D2_valid = D2
    r_bins = 10 ** np.arange(0.3, np.log(maxlength), 0.1)
    D2_avg = sf.average_structure_function(D2_valid, r_bins, mode='mean')
    D2_avg = D2_avg.assign_coords(lat=seg['lat'].mean(),
                                  lon=seg['lon'].mean(),
                                  mission=seg.attrs['mission'],
                                  segment_number=seg.attrs['segment_number'],
time=pd.to_datetime(np.mean(pd.to_numeric(seg['time'].data)))
                                 )    
    D2_list.append(D2_avg)
D2_all = xr.concat(D2_list, dim='segment')

D2_all.to_netcdf(args.outfile)