
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

Activate the UI environment::

   source setup.sh

and compile::

   make


Generating events and images
----------------------------

For example, to generate Pythia events::

   generate --batch long --random-state 100 w.config --events 1000 --output w_100.h5 --params "PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320"

See the ``pythia:`` rule in the Makefile for a complete sample generation.

To then reconstruct and cluster events in HDF5 files (Pythia events) or HepMC files::

   cluster --batch long --delphes --jet-size 1.0 --subjet-size 0.3 --subjet-pt-min-fraction 0.05 --delphes-config delphes_card_ATLAS_NoFastJet.tcl w_100.h5
   cluster --batch long --delphes --jet-size 1.0 --subjet-size 0.3 --subjet-pt-min-fraction 0.05 --delphes-config delphes_card_ATLAS_PileUp_NoFastJet.tcl --suffix pileup w_100.h5

The first command above runs the reconstruction and clustering without pileup.
The second command turns on pileup. These commands will create the files
``w_100_j1p0_sj0p30_delphes_jets.h5`` and ``w_100_j1p0_sj0p30_delphes_jets_pileup.h5``

Create HDF5 files containing jet images with ``imgify``::

   imgify --batch long w_100_j1p0_sj0p30_delphes_jets_pileup.h5

That will create a file named
``w_100_j1p0_sj0p30_delphes_jets_pileup_images.h5``.

Merge multiple images files (from different jobs using different random seeds, for example) into one dataset with the ``sample merge`` command::

   sample --batch long merge -o w_j1p0_sj0p30_delphes_jets_images.h5 w_[0-9]*delphes_jets_images.h5

Finally, to apply a network on datasets of images::

   apply-network --batch long /coepp/cephfs/mel/edawe/deepjets/models/delphes_m_50_110/delphes_nozoom_a34d582c72fe4d438ae37f2409a62c9c_lr0.001_bs100 w_j1p0_sj0p30_delphes_jets_* qcd_j1p0_sj0p30_delphes_jets_*

That will create a ``*_proba.h5`` file containing network scores per input images dataset file.


Checking consistency with reference PYTHIA event
------------------------------------------------

After making changes to package versions, the event generation code, etc the
events produced for a fixed random seed might begin to differ. Check for
differences with a reference event as follows::

   ./generate qcd.config --write-hepmc --events 1 --random-state 101
   diff qcd.hepmc qcd.hepmc.reference


Running Herwig
--------------

On the Melbourne IU, Herwig has its own environment since it didn't seem to
play nice with the default environment from ``source setup.sh``.
So run Herwig in a fresh terminal after the following::

   source /data/edawe/public/software/hep/herwig/bin/activate
