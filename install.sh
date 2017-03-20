
if [ -z ${DEEPJETS_SFT_DIR+x} ]; then
    echo "first run \"source setup.sh\""
    exit 1
fi

PREFIX=$DEEPJETS_SFT_DIR
mkdir -p $PREFIX/src
cd $PREFIX/src

fastjet_version=3.2.1
fjcontrib_version=1.025
pythia_version=8219
vincia_version=2.0.01
hepmc_version=2.06.09
delphes_version=3.4.0

if [ ! -d fastjet-${fastjet_version} ]; then
    wget http://fastjet.fr/repo/fastjet-${fastjet_version}.tar.gz
    tar xfz fastjet-${fastjet_version}.tar.gz
fi

if [ ! -d fjcontrib-${fjcontrib_version} ]; then
    wget http://fastjet.hepforge.org/contrib/downloads/fjcontrib-${fjcontrib_version}.tar.gz
    tar xfz fjcontrib-${fjcontrib_version}.tar.gz
fi

if [ ! -d pythia${pythia_version} ]; then
    wget http://home.thep.lu.se/~torbjorn/pythia8/pythia${pythia_version}.tgz
    tar xfz pythia${pythia_version}.tgz
fi

if [ ! -d vincia-${vincia_version} ]; then
    wget http://www.hepforge.org/archive/vincia/vincia-2.0.01.tgz
    tar xfz vincia-${vincia_version}.tgz
fi

if [ ! -d HepMC-${hepmc_version} ]; then
    wget http://lcgapp.cern.ch/project/simu/HepMC/download/HepMC-${hepmc_version}.tar.gz
    tar xfz HepMC-${hepmc_version}.tar.gz
fi

if [ ! -d Delphes-${delphes_version} ]; then
    wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-${delphes_version}.tar.gz
    tar xfz Delphes-${delphes_version}.tar.gz
fi

cd fastjet-${fastjet_version}
make clean
./configure --prefix=$PREFIX --enable-cgal --enable-allcxxplugins --enable-all-plugins
make -j2
make install
cd ..

cd fjcontrib-${fjcontrib_version}
make clean
./configure --prefix=$PREFIX --fastjet-config=$PREFIX/bin/fastjet-config
make -j2
make install
make fragile-shared-install
cd ..

cd pythia${pythia_version}
make clean
./configure --prefix=$PREFIX --enable-shared --with-lhapdf6=$PREFIX --with-lhapdf6-plugin=lhapdf6
make -j2
make install
chmod +x $PREFIX/bin/pythia8-config
cd ..

cd vincia-${vincia_version}
make clean
./configure --prefix=$PREFIX --enable-shared
make -j2
make install
cd ..

cd HepMC-${hepmc_version}
make clean
./configure --prefix=$PREFIX --with-momentum=GEV --with-length=MM
make -j2
make install
cd ..

cd Delphes-${delphes_version}
make clean
./configure
make -j2
cp libDelphes.so libDelphesNoFastJet.so $PREFIX/lib/
mkdir $PREFIX/include/Delphes
cp -r modules/ $PREFIX/include/Delphes
cp -r classes/ $PREFIX/include/Delphes
mkdir $PREFIX/share/Delphes
cp -r cards/ $PREFIX/share/Delphes
cp -r external/ExRootAnalysis $PREFIX/include/
cd ..
