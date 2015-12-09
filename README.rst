
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
