#!/usr/bin/env bash

FOLDER=test_stochax
COMMAND="python suite.py --test"

export N_JOBS="1"
export LONG_TEST="False"

echo "#########################################"
echo "### Execute tests for stochax package ###"
echo "#########################################"


$COMMAND $FOLDER/test_base.py
$COMMAND $FOLDER/test_calibration_results.py
$COMMAND $FOLDER/test_core.py
$COMMAND $FOLDER/test_stochastic_processes.py

echo "Done!"
