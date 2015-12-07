export PYTHIADIR=/usr/local
export FASTJETDIR=/usr/local
python setup.py build_ext --inplace
python test.py
