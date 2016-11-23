# deterine path to this script
# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE_DEEPJETS_SETUP="${BASH_SOURCE[0]:-$0}"

DIR_DEEPJETS_SETUP="$( dirname "$SOURCE_DEEPJETS_SETUP" )"
while [ -h "$SOURCE_DEEPJETS_SETUP" ]
do
  SOURCE_DEEPJETS_SETUP="$(readlink "$SOURCE_DEEPJETS_SETUP")"
  [[ $SOURCE_DEEPJETS_SETUP != /* ]] && SOURCE_DEEPJETS_SETUP="$DIR_DEEPJETS_SETUP/$SOURCE_DEEPJETS_SETUP"
  DIR_DEEPJETS_SETUP="$( cd -P "$( dirname "$SOURCE_DEEPJETS_SETUP"  )" && pwd )"
done
DIR_DEEPJETS_SETUP="$( cd -P "$( dirname "$SOURCE_DEEPJETS_SETUP" )" && pwd )"

if [ -z ${DEEPJETS_SFT_DIR+x} ]; then
    # change the following default path if desired
    # otherwise set DEEPJETS_SFT_DIR yourself elsewhere before sourcing this
    export DEEPJETS_SFT_DIR=/data/edawe/public/software/hep
fi

if [ -e ${DEEPJETS_SFT_DIR}/setup.sh ]; then
    # place any environment customizations in a setup.sh located
    # in DEEPJETS_SFT_DIR (i.e. python and ROOT setup)
    source ${DEEPJETS_SFT_DIR}/setup.sh
fi

export PYTHIA8=${DEEPJETS_SFT_DIR}
export PYTHIA8DATA=${DEEPJETS_SFT_DIR}/share/Pythia8/xmldoc
export VINCIADATA=${DEEPJETS_SFT_DIR}/share/Vincia/xmldoc

# you should not need to edit below this line
export PATH=${DIR_DEEPJETS_SETUP}:${DEEPJETS_SFT_DIR}/bin:${PATH}
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${LD_LIBRARY_PATH}
fi

export PYTHONPATH=${DIR_DEEPJETS_SETUP}:${PYTHONPATH}
export DEEPJETS_DIR=$DIR_DEEPJETS_SETUP
