#!/usr/bin/env python
# Useful python module
import xarray as xr
import os
import argparse
import matplotlib.pyplot as plt

# Local modules
import plot
import quality as qc

# Define the parser for input parameters

description = ("Perform data segmentation"
              )
parser = argparse.ArgumentParser(description=description)
parser.add_argument('inpath', help="Input path")
parser.add_argument('outpath', help="Output path")
parser.add_argument('--region', choices=['VAUB', 'SCAL', 'ECAL', 'NCAL'], 
                                required=True,
                    help="Region to study")
parser.add_argument('--min_ship_velocity', default=3,
                    help='Minimum cruise speed allowed for the ship')
parser.add_argument('--max_break', default='15min',
                    help='Maximum break allowed in minutes')
parser.add_argument('--max_heading_rate', default=5,
                    help='Maximum break allowed')
parser.add_argument('--min_length', default=100,
                    help='Minimum length of segments')
parser.add_argument('--min_observations', default=50,
                    help='Minimum number of observations in segments')

args = parser.parse_args()


# Definition of the different polygon
from shapely.geometry import Polygon
poly_regions = dict()
poly_regions['SCAL'] = Polygon([[162, -23], [171, -23], [171, -27], [162, -27]])
poly_regions['ECAL'] = Polygon([[169, -20.5], [169, -24], [180, -24], [180, -20.5]])
poly_regions['NCAL'] = Polygon([[169, -20], [166, -20], [162, -18], [162, -12], 
                                [166, -12], [166, -15]])
poly_regions['VAUB'] = Polygon([[167.1, -22], [165.5, -21], [164.5, -20], [165.5, -20], 
                                [167, -21], [168.5, -22]])


def data_segmentation(input_path, output_path, mission_name,
                           min_ship_velocity=3, max_break='10min',   
                           max_heading_rate=5, min_length=100, 
                           min_observations=50, poly=None):
    """
    Perform a segmentation on the ship data
    """
    print("Processing mission %s." % mission_name)  
    try:
        ds = xr.open_dataset('%s/%s.nc' % (input_path, mission_name))
        segments = qc.get_segments(ds, min_uship=min_ship_velocity,
                                   max_heading_rate=max_heading_rate,
                                   poly=poly, attrs={'mission': mission_name})
        if segments:
            concat_segments = qc.concatenate_segments(segments, 
                                                      max_break=max_break,                                                      
                                                      max_heading_rate=max_heading_rate)
            valid_segments = qc.get_valid_segments(concat_segments, 
                                                   min_length=min_length, 
                                                   min_observations=min_observations)
            if valid_segments:
                qc.save_segments_to_netcdf(valid_segments, output_path, mission_name)
                print("Segmentation of mission %s sucessfully saved." % mission_name)
                plot.monitor_segments(ds, valid_segments, output_path, mission_name)
                plot.plot_segments(raw_data=ds, segments=valid_segments, 
                                   output_path=output_path, name=mission_name, 
                                   lat_min=-26, lat_max=-16, 
                                   lon_min=160, lon_max=175)
                return valid_segments
    except (KeyError, IndexError, ValueError):
        print("Error processing mission %s." % mission_name)  
        

filenames = sorted(os.listdir(args.inpath))
missions = [f[:-3] for  f in filenames]
for mission in missions:
    valid_segments = data_segmentation(args.inpath, args.outpath, mission,
                                       min_ship_velocity=float(args.min_ship_velocity),
                                       max_break=args.max_break,
                                       max_heading_rate=float(args.max_heading_rate), 
                                       min_length=int(args.min_length),
                                       min_observations=int(args.min_observations),
                                       poly=poly_regions[args.region])
    
# Use imagemagick to merge all the plots generated during the segmentation
os.system("convert %s/*/*segment_map.png %s/segment_maps.pdf" % (args.outpath, args.outpath))
os.system("convert %s/*/*ship_velocity_and_heading.png %s/ship_velocity_and_heading.pdf" % (args.outpath, args.outpath))