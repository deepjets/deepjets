#!/usr/bin/env python

from __future__ import print_function

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "numpy cannot be imported. numpy must be installed "
        "prior to installing deepjets")

try:
    # try to use setuptools if installed
    from pkg_resources import parse_version, get_distribution
    from setuptools import setup, Extension
    if get_distribution('setuptools').parsed_version < parse_version('0.7'):
        # before merge with distribute
        raise ImportError
except ImportError:
    # fall back on distutils
    from distutils.core import setup
    from distutils.extension import Extension

import os
import sys
import subprocess
from glob import glob
from Cython.Build import cythonize

# Prevent setup from trying to create hard links
# which are not allowed on AFS between directories.
# This is a hack to force copying.
try:
    del os.link
except AttributeError:
    pass

local_path = os.path.dirname(os.path.abspath(__file__))
# setup.py can be called from outside the root_numpy directory
os.chdir(local_path)
sys.path.insert(0, local_path)

try:
    DEEPJETS_SFT_DIR=os.environ['DEEPJETS_SFT_DIR']
except KeyError:
    raise RuntimeError("first run \"source setup.sh\"")


def root_flags(root_config='root-config'):
    root_cflags = subprocess.Popen(
        [root_config, '--cflags'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen(
        [root_config, '--libs'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    if sys.version > '3':
        root_cflags = root_cflags.decode('utf-8')
        root_ldflags = root_ldflags.decode('utf-8')
    return root_cflags.split(), root_ldflags.split()

rootsys = os.getenv('ROOTSYS', None)
if rootsys is not None:
    try:
        root_config = os.path.join(rootsys, 'bin', 'root-config')
        root_cflags, root_ldflags = root_flags(root_config)
    except OSError:
        raise RuntimeError(
            "ROOTSYS is {0} but running {1} failed".format(
                rootsys, root_config))
else:
    try:
        root_cflags, root_ldflags = root_flags()
    except OSError:
        raise RuntimeError(
            "root-config is not in PATH and ROOTSYS is not set. "
            "Is ROOT installed correctly?")

libdeepjets = Extension(
    'deepjets._libdeepjets',
    sources=['deepjets/src/_libdeepjets.pyx'],
    depends=glob('deepjets/src/*.h'),
    language='c++',
    include_dirs=[
        np.get_include(),
        'deepjets/src',
        '/usr/local/include',
        os.path.join(DEEPJETS_SFT_DIR, 'include'),
        os.path.join(DEEPJETS_SFT_DIR, 'include', 'Delphes'),
    ],
    library_dirs=[
        '/usr/local/lib',
        os.path.join(DEEPJETS_SFT_DIR, 'lib'),
    ],
    libraries=('pythia8 fastjet fastjetcontribfragile '
               'CGAL gmp HepMC DelphesNoFastJet').split(),
    extra_compile_args=root_cflags + [
        '-Wno-unused-function',
        '-Wno-write-strings',
    ],
    extra_link_args=root_ldflags)

setup(
    name='deepjets',
    version='0.0.1',
    packages=['deepjets'],
    package_data={
        'deepjets': ['testdata/*.hepmc'],
    },
    ext_modules=cythonize([libdeepjets]),
)
