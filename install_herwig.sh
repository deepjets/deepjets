
if [ -z ${DEEPJETS_SFT_DIR+x} ]; then
    echo "first run \"source setup.sh\""
    exit 1
fi

PREFIX=$DEEPJETS_SFT_DIR
mkdir -p $PREFIX/src
cd $PREFIX/src

curl -O https://herwig.hepforge.org/hg/bootstrap/raw-file/published/herwig-bootstrap
./herwig-bootstrap --with-fastjet $PREFIX --with-hepmc $PREFIX $PREFIX
