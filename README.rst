
deepjets: Deep Learning Jet Images
==================================

Reimplementing http://arxiv.org/abs/1511.05190

Install the following (more detailed instructions on the way):

* PYTHIA: http://home.thep.lu.se/Pythia/
* FastJet: http://fastjet.fr/
* CGAL: http://www.cgal.org/
* GMP: https://gmplib.org/

Install required Python packages::

   pip install --user numpy scipy matplotlib scikit-image cython

Build and run the test script with::

   export PYTHIADIR=/usr/local
   export FASTJETDIR=/usr/local
   python setup.py build_ext --inplace
   python test.py
