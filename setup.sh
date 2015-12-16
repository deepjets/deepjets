# change the following path if desired
export DEEPJETS_SFT_DIR=/data/edawe/public/software/hep # on the UI

# you should not need to edit below this line
export PATH=${DEEPJETS_SFT_DIR}/bin:${PATH}
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${LD_LIBRARY_PATH}
fi
