#!/usr/bin/env bash
set -eu

FOLDER=test_stochax

export N_JOBS="1"
export LONG_TEST="False"

echo "#########################################"
echo "### Execute tests for stochax package ###"
echo "#########################################"


coverage run suite.py --test $FOLDER
coverage report -m

echo ""
echo ""
echo "Done!"
