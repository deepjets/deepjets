
if [ -z ${DEEPJETS_SFT_DIR+x} ]; then
    echo "first run \"source setup.sh\""
    exit 1
fi

PREFIX=$DEEPJETS_SFT_DIR
mkdir -p $PREFIX/src
cd $PREFIX/src

if [ ! -d fastjet-3.1.3 ]; then
    wget http://fastjet.fr/repo/fastjet-3.1.3.tar.gz
    tar xvfz fastjet-3.1.3.tar.gz
fi

if [ ! -d pythia8212 ]; then
    wget http://home.thep.lu.se/~torbjorn/pythia8/pythia8212.tgz
    tar xvfz pythia8212.tgz
fi

cd fastjet-3.1.3
make clean
./configure --prefix=$PREFIX --enable-cgal
make -j2
make install
cd ..

cd pythia8212
make clean
./configure --prefix=$PREFIX
make -j2
make install
chmod +x $PREFIX/bin/pythia8-config
cd ..
