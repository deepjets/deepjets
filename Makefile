# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)

all: clean inplace

clean-pyc:
	@find . -name "*.pyc" -exec rm {} \;

clean-so:
	@find deepjets -name "*.so" -exec rm {} \;

clean-build:
	@rm -rf build

clean: clean-build clean-pyc clean-so

in: inplace # just a shortcut
inplace:
	@$(PYTHON) setup.py build_ext -i

events:
	./dump wprime.config --events 1000000 --cut-on-pdgid 24 --pt-min 100 &
	./dump qcd.config --events 1000000 &
