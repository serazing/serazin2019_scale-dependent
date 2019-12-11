# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geometry as geom
import cartopy.crs as ccrs
import os

def get_segments(ds, min_uship=3., max_heading_rate=5, poly=None, attrs={}):
	"""
	Make a list of segments from an ADCP dataset to consider only portions
	where the ship meet velocity and heading rates requirements

	Parameters
	----------
	ds : xarray.Dataset
		The original dataset
	min_uship : double, optional
		Minimum ship velocity for which the segments will be considered.
		Default is 3 m/s.
	max_heading_rate : double, optional
		Maximum heading variation compared to previous measurement.
		Default is 5 degree.
    poly : BaseGeometry
        A polygon defined using shapely, which defined a
	attrs : dictionary-like
		Define the global attributes of the dataset

	Returns
	-------
	segments : list of xarray.Dataset
		Return the values where the segments meet the required conditions
	"""
	segments = []
	if poly is not None:
		cond_region = geom.inpolygon(ds, poly)
	else:
		cond_region = 1.
	# Compute absolute ship velocity and test against the minimum velocity
	uship = xr.ufuncs.sqrt(ds['uship'] ** 2 + ds['vship'] ** 2)
	cond_velocity = (uship > min_uship)
	# Compute the heading time rate and compare the maximum value allowed
	heading_rate = abs(ds['heading'].diff(dim='time'))
	ds = ds.where((heading_rate < max_heading_rate) & cond_velocity &
	              cond_region)
	# Useful numpy.ma function to get the slices corresponding to unmasked data
	slices = np.ma.clump_unmasked(np.ma.masked_invalid(ds['uship'].data))
	segment_counter = 1
	for sl in slices:
		seg = ds.isel(time=sl)
		seg.attrs = attrs
		seg.attrs['segment_number'] = segment_counter
		segments.append(seg)
		segment_counter += 1
	return segments


def _compute_time_between_segments(segments):
	"""
	Compute the time break separating different segments, and add this
	information o each segment in the attribute
	`time_break_with_previous_segment`

	Parameters
	----------
	segments : list of Datasets
		List of segments to use
	"""
	time_break = []
	previous_segment = segments[0]
	for seg in segments:
		delta_time = pd.to_timedelta((seg['time'][0] -
                                      previous_segment['time'][-1]).data)
		time_break.append(delta_time)
		previous_segment = seg
	return time_break


def _compute_heading_difference_between_segments(segments):
	"""
	Compute the heading difference between segments, and add this
	information o each segment in the attribute
	`heading_difference_with_previous_segment`
	"""
	heading_difference = []
	previous_segment_heading = segments[0]['heading'].mean()
	for seg in segments:
		segment_heading = seg['heading'].mean()
		delta_heading = abs(previous_segment_heading - segment_heading)
		heading_difference.append(delta_heading.data)
		previous_segment_heading = segment_heading
	return heading_difference


def concatenate_segments(segments, max_break='30m', max_heading_rate=5):
	"""
	Concatenate consecutive segments if they meet the requirements defined by
	the maximum ship break allowed and the maximum heading deviation allowed.

	Parameters
	---------
	segments : list of Datasets
		The list of segments to compare between each other
	max_break : str, optional
		The maximum time break allowed to concatenate.
	max_heading_rate :
		The maximum heading deviation allowed to concatenate.

	Returns
	-------
	concatenated_segments :  list of Datasets
		A list of segments of size smaller than the input
	"""
	time_break = _compute_time_between_segments(segments)
	heading_dev = _compute_heading_difference_between_segments(segments)
	max_break = pd.to_timedelta(max_break)
	concatenated_segments = []
	current_segment = [segments[0]]
	for i in range(1, len(segments)):
		seg = segments[i]
		if time_break[i] <= max_break and heading_dev[i] <= max_heading_rate:
			current_segment.append(seg)
		else:
			concatenated_segments.append(xr.concat(current_segment, dim='time'))
			current_segment = [seg]
			# Do not forget to concatenate the last segment
	concatenated_segments.append(xr.concat(current_segment, dim='time'))
	return concatenated_segments


def get_segment_length(segment):
	"""
	Compute the spatial length of segment

	Parameters
	----------
	segment : xarray.Dataset
		A segment defined by a Dataset

	Returns
	-------
	length : float
		The spatial length of the segment
	"""
	y_stop, x_stop = geom.latlon2yx(segment['lat'][-1], segment['lon'][-1])
	y_start, x_start = geom.latlon2yx(segment['lat'][0], segment['lon'][0])
	length = 1e-3 * xr.ufuncs.sqrt(
		(y_stop - y_start) ** 2 + (x_stop - x_start) ** 2)
	return length


def get_valid_segments(segments, min_length=50, min_observations=20):
	"""
	Get only valid segments by checking their spatial length and the minimum
	of observations included

	Paramaters
	----------
	segments : list of xarray.Dataset
		The list of segments to evaluate
	min_length : float, optional
		The minimum spatial length of a segment
	min_observations : int, optional
		The minimum number of observations in a segment

	Returns
	-------
	valid_segments : list of xarray.Dataset
		The list of segments that meet the requirements
	"""
	valid_segments = []
	for seg in segments:
		if (get_segment_length(seg) >= min_length and
			seg.sizes['time'] >= min_observations):
			valid_segments.append(seg)
	return valid_segments



def save_segments_to_netcdf(segments, output_path, mission_name=''):
    full_output_path = "%s/%s/" % (output_path, mission_name)
    if not os.path.isdir(full_output_path):
        os.makedirs(full_output_path)
    output_filenames = [full_output_path + mission_name +
                        '_segment_%02d.nc' % (ns + 1)  for ns in range(len(segments))]
    xr.save_mfdataset(segments, output_filenames)


def open_segments_from_netcdf(input_path):
    import glob
    list_of_path = sorted(glob.glob(input_path + "*/*_segment_*.nc"))
    return [xr.open_dataset(path, autoclose=True) for path in list_of_path]