#!/usr/bin/env bash

FOLDER=test_stochax
COMMAND="python suite.py --test"

export N_JOBS="1"

echo ">>>>>>>>>>>>"
echo ">>> stochax"
echo ">>>>>>>>>>>>"


$COMMAND $FOLDER/test_calibration_results.py
$COMMAND $FOLDER/test_core.py
$COMMAND $FOLDER/test_stochastic_processes.py

echo "Done!"
