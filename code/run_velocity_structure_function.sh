#!/usr/bin/env bash


REGION='STCC'
INPUT_PATH='/data/OBS/ADCP/'$REGION
OUTPUT_FILE='/data/RESULTS/STRUCTURE_FUNCTIONS/VELOCITY/'$REGION'_velocity_structure_function.nc'

./compute_velocity_structure_functions.py $INPUT_PATH $OUTPUT_FILE


