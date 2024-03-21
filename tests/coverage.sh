#!/usr/bin/env bash

coverage run suite.py --test test_stochax/
coverage report -m
coverage html
