#!/usr/bin/env bash


REGION='NCAL'

#INPUT_PATH='/data/OBS/ADCP/CLEAN'
#OUTPUT_PATH='/data/OBS/ADCP/'$REGION
INPUT_PATH='/data/OBS/TSG/LEGOS_CLEAN'
OUTPUT_PATH='/data/OBS/TSG/'$REGION

MAX_HEADING_RATE=5
MIN_SHIP_VELOCITY=4
MIN_LENGTH=100

rm -rf $OUTPUT_PATH/
./data_segmentation.py $INPUT_PATH $OUTPUT_PATH --region $REGION --max_heading_rate $MAX_HEADING_RATE --min_ship_velocity $MIN_SHIP_VELOCITY --min_length $MIN_LENGTH

