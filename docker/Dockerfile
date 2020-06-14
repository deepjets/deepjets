FROM ubuntu:16.04

# image config
LABEL name="deepjets"
LABEL version="0.1"

# basic environment variables
ENV PATH $PATH:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:/root/deepjets
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV LIBPATH $LIBPATH:/usr/local/lib
ENV PYTHONPATH $PYTHONPATH:/usr/local/lib:/usr/local/lib64/python2.7/site-packages
ENV PKG_CONFIG_PATH $PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
ENV CMAKE_MODULE_PATH $CMAKE_MODULE_PATH;/usr/local/etc/cmake
ENV MANPATH $MANPATH:/usr/local/man
ENV DEEPJETS_SFT_DIR /root/software

# setup software
RUN apt-get -y update; apt-get clean
RUN apt-get -y install build-essential libcgal-dev libcgal11v5 libgmp-dev libgmp10 libhdf5-dev \
    python python-dev nano git wget tcl tk expect subversion libapache2-svn; apt-get clean
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install root-system; apt-get clean
RUN wget https://bootstrap.pypa.io/get-pip.py; python get-pip.py; rm get-pip.py
RUN pip install cython numpy scipy matplotlib scikit-image h5py pydot dask cloudpickle toolz \
    blessings progressbar2 scikit-learn pyparsing joblib
RUN pip install -U https://github.com/Theano/Theano/zipball/master
RUN pip install -U https://github.com/fchollet/keras/zipball/master

# fetch the latest code
RUN git clone https://github.com/deepjets/deepjets.git /root/deepjets
WORKDIR /root/deepjets

# setup additional software
RUN /bin/bash -c "source /root/deepjets/setup.sh && ./install.sh"
RUN /bin/bash -c "source /root/deepjets/setup.sh && CFLAGS='-std=c++11' make -j"
RUN echo "source /root/deepjets/setup.sh" >> /root/.bashrc

# default command
CMD bash
