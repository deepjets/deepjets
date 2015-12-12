
deepjets: Deep Learning Jet Images
==================================

Reimplementing http://arxiv.org/abs/1511.05190

Install `CGAL <http://www.cgal.org/>`_ and `GMP <https://gmplib.org/>`_::

   sudo apt-get install libcgal-dev libcgal11v5 libgmp-dev libgmp10

on Fedora::

   sudo dnf install gmp.x86_64 gmp-devel.x86_64 CGAL.x86_64 CGAL-devel.x86_64

or on Mac OS::

   brew install cgal gmp boost

Install `PYTHIA <http://home.thep.lu.se/Pythia/>`_ and
`FastJet <http://fastjet.fr/>`_ with the ``install.sh`` script::

   ./install.sh

If you don't have pip installed, do the following::

   curl -O https://bootstrap.pypa.io/get-pip.py
   python get-pip.py --user

If this isn't in your .bashrc already, add it::

   export PATH=~/.local/bin${PATH:+:$PATH}

Install HDF5 and h5py (we use this to store the jet images and neural nets)::

   sudo apt-get install libhdf5-dev
   pip install --user h5py

Install required Python packages::

   pip install --user cython numpy scipy matplotlib scikit-image keras

Build and run the test script with::

   export PYTHIA_DIR=/usr/local
   export FASTJET_DIR=/usr/local
   python setup.py build_ext --inplace
   python test.py
