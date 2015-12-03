# Makefile is a part of the PYTHIA event generator.
# Copyright (C) 2014 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, September 2014.
#
# This is is the Makefile used to build PYTHIA examples on POSIX systems.
# Example usage is:
#     make main01
# For help using the make command please consult the local system documentation,
# i.e. "man make" or "make --help".

################################################################################
# VARIABLES: Definition of the relevant variables from the configuration script.
################################################################################

# Include the configuration.
-include Makefile.inc

# Handle GZIP support.
ifeq ($(GZIP_USE),true)
  CXX_COMMON+= -DGZIPSUPPORT
  CXX_COMMON+= -L$(BOOST_LIB) -Wl,-rpath $(BOOST_LIB) -lboost_iostreams
  CXX_COMMON+= -L$(GZIP_LIB) -Wl,-rpath $(GZIP_LIB) -lz
endif

# Check distribution (use local version first, then installed version).
#ifneq ("$(wildcard ../../../lib/libpythia8.a)","")
  PREFIX_LIB=/Users/barney800/Tools/Pythia8/lib
  PREFIX_INCLUDE=/Users/barney800/Tools/Pythia8/include
#endif
CXX_COMMON:=-I$(PREFIX_INCLUDE) $(CXX_COMMON) -Wl,-rpath $(PREFIX_LIB) -ldl

################################################################################
# RULES: Definition of the rules used to build the PYTHIA examples.
################################################################################

# Rules without physical targets (secondary expansion for specific rules).
.SECONDEXPANSION:
.PHONY: all clean

# All targets (no default behavior).
all:
	@echo "Usage: make mainXX"

# The Makefile configuration.
Makefile.inc:
	$(error Error: PYTHIA must be configured, please run "./configure"\
                in the top PYTHIA directory)

# PYTHIA libraries.
$(PREFIX_LIB)/libpythia8.a :
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory)

# Examples without external dependencies.
main% : main%.cc $(PREFIX_LIB)/libpythia8.a
	$(CXX) $^ -o $@ $(CXX_COMMON)

# User-written examples for tutorials, without external dependencies.
mymain% : mymain%.cc $(PREFIX_LIB)/libpythia8.a
	$(CXX) $^ -o $@ $(CXX_COMMON)

# HEPMC2.
mymainH main41 main42 main61 main62 main85 main86 main87 main88 main89: $$@.cc\
	$(PREFIX_LIB)/libpythia8.a
ifeq ($(HEPMC2_USE),true)
	$(CXX) $^ -o $@ -I$(HEPMC2_INCLUDE) $(CXX_COMMON)\
	 -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC
else
	@echo "Error: $@ requires HEPMC2"
endif

# FASTJET3.
main71 main72 mymain1 WprimeJetGen: $$@.cc $(PREFIX_LIB)/libpythia8.a
ifeq ($(FASTJET3_USE),true)
	$(CXX) $^ -o $@ -I$(FASTJET3_INCLUDE) $(CXX_COMMON)\
	 -L$(FASTJET3_LIB) -Wl,-rpath $(FASTJET3_LIB) -lfastjet
else
	@echo "Error: $@ requires FASTJET3"
endif

# FASTJET3 and HEPMC2.
main81 main82 main83 main84: $$@.cc $(PREFIX_LIB)/libpythia8.a
ifeq ($(FASTJET3_USE)$(HEPMC2_USE),truetrue)
	$(CXX) $^ -o $@ -I$(FASTJET3_INCLUDE) -I$(HEPMC2_INCLUDE) $(CXX_COMMON)\
	 -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC\
	 -L$(FASTJET3_LIB) -Wl,-rpath $(FASTJET3_LIB) -lfastjet
else
	@echo "Error: $@ requires FASTJET3 and HEPMC2"
endif

# ROOT (turn off all warnings for readability).
main91: $$@.cc $(PREFIX_LIB)/libpythia8.a
ifeq ($(ROOT_USE),true)
	$(CXX) $^ -o $@ -w -I$(ROOT_INCLUDE) $(CXX_COMMON)\
	 -Wl,-rpath $(ROOT_LIB) `$(ROOT_BIN)root-config --glibs`
else
	@echo "Error: $@ requires ROOT"
endif
main92: $$@.cc $$@.h $$@LinkDef.h $(PREFIX_LIB)/libpythia8.a
ifeq ($(ROOT_USE),true)
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(ROOT_LIB);\
	 $(ROOT_BIN)rootcint -f $@Dct.cc -c -I$(PREFIX_INCLUDE) $@.h $@LinkDef.h
	$(CXX) $@Dct.cc $^ -o $@ -w -I$(ROOT_INCLUDE) $(CXX_COMMON)\
	 -Wl,-rpath $(ROOT_LIB) `$(ROOT_BIN)root-config --glibs`
else
	@echo "Error: $@ requires ROOT"
endif

# Clean.
clean:
	@rm -f main[0-9][0-9]; rm -f out[0-9][0-9]; rm -f weakbosons.lhe;\
	rm -f mymain[0-9][0-9]; rm -f myout[0-9][0-9]; rm -f hist.root;\
	rm -f *~; rm -f \#*; rm -f core*; rm -f *Dct.*
