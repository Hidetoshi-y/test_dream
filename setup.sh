#!/bin/bash
apt-get update
apt-get install liblapack-dev
apt-get install gfortran
pip install keras==2.0.8 || exit 1
pip install numpy || exit 1
pip install tensorflow'>='1.14.0 || exit 1
pip install scipy==1.1.0 || exit 1