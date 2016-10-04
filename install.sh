
if [ -z ${DEEPJETS_SFT_DIR+x} ]; then
    echo "first run \"source setup.sh\""
    exit 1
fi

PREFIX=$DEEPJETS_SFT_DIR
mkdir -p $PREFIX/src
cd $PREFIX/src

if [ ! -d fastjet-3.1.3 ]; then
    wget http://fastjet.fr/repo/fastjet-3.1.3.tar.gz
    tar xfz fastjet-3.1.3.tar.gz
fi

if [ ! -d fjcontrib-1.021 ]; then
    wget http://fastjet.hepforge.org/contrib/downloads/fjcontrib-1.021.tar.gz
    tar xfz fjcontrib-1.021.tar.gz
fi

if [ ! -d pythia8219 ]; then
    wget http://home.thep.lu.se/~torbjorn/pythia8/pythia8219.tgz
    tar xfz pythia8219.tgz
fi

if [ ! -d HepMC-2.06.09 ]; then
    wget http://lcgapp.cern.ch/project/simu/HepMC/download/HepMC-2.06.09.tar.gz
    tar xfz HepMC-2.06.09.tar.gz
fi

if [ ! -d Delphes-3.3.2 ]; then
    wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.3.2.tar.gz
    tar xfz Delphes-3.3.2.tar.gz
fi

cd fastjet-3.1.3
make clean
./configure --prefix=$PREFIX --enable-cgal --enable-allcxxplugins
make -j2
make install
cd ..

cd fjcontrib-1.021
make clean
./configure --prefix=$PREFIX --fastjet-config=$PREFIX/bin/fastjet-config
make -j2
make install
make fragile-shared-install
cd ..

cd pythia8219
make clean
#./configure --prefix=$PREFIX --enable-shared
./configure --prefix=$PREFIX --enable-shared --enable-debug --with-lhapdf6=$PREFIX --with-lhapdf6-plugin=lhapdf6
make -j2
make install
chmod +x $PREFIX/bin/pythia8-config
cd ..

cd HepMC-2.06.09
make clean
./configure --prefix=$PREFIX --with-momentum=GEV --with-length=MM
make -j2
make install
cd ..

cd Delphes-3.3.2
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

curl -O https://herwig.hepforge.org/hg/bootstrap/raw-file/published/herwig-bootstrap
./herwig-bootstrap --with-fastjet $PREFIX --with-hepmc $PREFIX $PREFIX
