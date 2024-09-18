#!/usr/bin/env bash
set -euo pipeline

FOLDER=test_stochax

export N_JOBS="1"
export LONG_TEST="False"

echo "#########################################"
echo "### Execute tests for stochax package ###"
echo "#########################################"


coverage run suite --test $FOLDER
coverage report -m

echo ""
echo ""
echo "Done!"
