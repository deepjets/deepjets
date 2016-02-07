
deepjets: Deep Learning Jet Images
==================================

Reimplementing http://arxiv.org/abs/1511.05190

Manual Installation
-------------------

See below for instructions on using the existing setup on the UI.

Install boost, `CGAL <http://www.cgal.org/>`_ and `GMP <https://gmplib.org/>`_.
On a Debian-based system (Ubuntu)::

   sudo apt-get install libcgal-dev libcgal11v5 libgmp-dev libgmp10

on an RPM-based system (Fedora)::

   sudo dnf install gmp.x86_64 gmp-devel.x86_64 CGAL.x86_64 CGAL-devel.x86_64

or on Mac OS::

   brew install cgal gmp boost wget

Set up the environment variables (always do this, even after installing for the
first time) by sourcing setup.sh. If desired, first change the software
installation path at the top of the file. Then run::

   source setup.sh

Install `PYTHIA <http://home.thep.lu.se/Pythia/>`_ and
`FastJet <http://fastjet.fr/>`_ and `HepMC <http://lcgapp.cern.ch/project/simu/HepMC/>`_
with the ``install.sh`` script::

   ./install.sh

If you don't have pip installed, do the following::

   curl -O https://bootstrap.pypa.io/get-pip.py
   python get-pip.py --user

If this isn't in your .bashrc already, add it::

   export PATH=~/.local/bin${PATH:+:$PATH}

Install HDF5 (we use this to store the jet images and neural nets).
On Debian-based systems::

   sudo apt-get install libhdf5-dev

On RPM-based systems::

   sudo dnf install hdf5.x86_64 hdf5-devel.x86_64

On Mac OS::

   brew install hdf5

Install required Python packages::

   pip install --user cython numpy scipy matplotlib scikit-image h5py pydot

Finally, install the latest Theano and keras::

   pip install --user -U https://github.com/Theano/Theano/zipball/master
   pip install --user -U https://github.com/fchollet/keras/zipball/master
   pip install pyparsing==1.5.7


Setting up the environment on the UI
------------------------------------

Activate Noel's UI environment::

   source /data/edawe/public/setup.sh
   source setup.sh


Compiling, and generating events and images
-------------------------------------------

Simply::

   make
   make events
   make images
