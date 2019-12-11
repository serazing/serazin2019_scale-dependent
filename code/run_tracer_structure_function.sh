#!/usr/bin/env bash


REGION='NCAL'
INPUT_PATH='/data/OBS/TSG/'$REGION
OUTPUT_FILE='/data/RESULTS/STRUCTURE_FUNCTIONS/SURFACE_TRACERS/'$REGION'_tracer_structure_function.nc'

./compute_tracer_structure_functions.py $INPUT_PATH $OUTPUT_FILE


