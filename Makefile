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

trailing-spaces:
	@find deepjets -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'
