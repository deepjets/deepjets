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

# change the following paths if desired
BASE=/data/edawe/public
if [ -e ${BASE} ]; then
  source ${BASE}/setup.sh
  export DEEPJETS_SFT_DIR=${BASE}/software/hep # on the UI
fi

# you should not need to edit below this line
export PATH=${DIR_DEEPJETS_SETUP}:${DEEPJETS_SFT_DIR}/bin:${PATH}
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${DEEPJETS_SFT_DIR}/lib:${LD_LIBRARY_PATH}
fi

export PYTHONPATH=${DIR_DEEPJETS_SETUP}:${PYTHONPATH}
