# PYTHIA configuration file.
# Generated on Tue 1 Dec 2015 12:38:07 AEDT with the user supplied options:
# --with-fastjet3=/Users/barney800/Tools/fastjet

# Install directory prefixes.
PREFIX_BIN=/usr/bin
PREFIX_INCLUDE=/usr/include
PREFIX_LIB=/usr/lib
PREFIX_SHARE=/usr/share/Pythia8

# Compilation flags (see ./configure --help for further documentation).
ENABLE_SHARED=false
CXX=g++
CXX_COMMON=-O2 -ansi -pedantic -W -Wall -Wshadow -fPIC
CXX_SHARED=-dynamiclib -flat_namespace -undefined suppress -Qunused-arguments
CXX_SONAME=-Wl,-dylib_install_name
LIB_SUFFIX=.dylib

# FASTJET3 configuration.
FASTJET3_USE=true
FASTJET3_BIN=/Users/barney800/Tools/fastjet/bin/
FASTJET3_INCLUDE=/Users/barney800/Tools/fastjet/include
FASTJET3_LIB=/Users/barney800/Tools/fastjet/lib

# HEPMC2 configuration.
HEPMC2_USE=false
HEPMC2_BIN=
HEPMC2_INCLUDE=
HEPMC2_LIB=

# HEPMC3 configuration.
HEPMC3_USE=false
HEPMC3_BIN=
HEPMC3_INCLUDE=
HEPMC3_LIB=

# LHAPDF5 configuration.
LHAPDF5_USE=false
LHAPDF5_BIN=
LHAPDF5_INCLUDE=
LHAPDF5_LIB=

# LHAPDF6 configuration.
LHAPDF6_USE=false
LHAPDF6_BIN=
LHAPDF6_INCLUDE=
LHAPDF6_LIB=

# ROOT configuration.
ROOT_USE=false
ROOT_BIN=
ROOT_INCLUDE=
ROOT_LIB=

# BOOST configuration.
BOOST_USE=false
BOOST_BIN=
BOOST_INCLUDE=
BOOST_LIB=

# GZIP configuration.
GZIP_USE=false
GZIP_BIN=
GZIP_INCLUDE=
GZIP_LIB=
