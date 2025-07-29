#!/bin/bash

#This script adds the usual suspects to a git commit.

#Add all MATLAB scripts
git add matlab_scripts/*.m

#Add all python scripts
git add jax_scripts/*.py
git add jax_scripts/lib/*.py

#Whatever Floquet thing I am studying
git add floquet.mat

#Add all solutions
git add solutions/Re100/*

git add github_add.sh 
