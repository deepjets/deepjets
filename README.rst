
deepjets: Deep Learning Jet Images
==================================

Reimplementing http://arxiv.org/abs/1511.05190

Install the following (more detailed instructions on the way):

* PYTHIA: http://home.thep.lu.se/Pythia/
* FastJet: http://fastjet.fr/
* CGAL: http://www.cgal.org/
* GMP: https://gmplib.org/

If you don't have pip installed, do the following::

   curl -O https://bootstrap.pypa.io/get-pip.py
   python get-pip.py --user

Install HDF5 and PyTables (we use this to store the jet images as numpy arrays)::

   sudo apt-get install libhdf5-dev
   pip install --user tables

Install required Python packages::

   pip install --user numpy scipy matplotlib scikit-image cython keras

Build and run the test script with::

   export PYTHIADIR=/usr/local
   export FASTJETDIR=/usr/local
   python setup.py build_ext --inplace
   python test.py
