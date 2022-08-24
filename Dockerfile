FROM ubuntu:18.04

ENV OC_ORIGIN="singnet"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y gnupg2 wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN echo "deb https://apt.kitware.com/ubuntu/ bionic main" | tee '/etc/apt/sources.list.d/cmake.list'

RUN apt-get update

RUN apt-get install -y \
      autoconf \
      build-essential \
      cmake \
      cython \
      gettext \
      guile-2.2-dev \
      libboost-all-dev \
      nano \
      python3-pip \
      unzip \
      git

RUN apt-get autoremove

RUN python3 -m pip install -U pip
RUN python3 -m pip install \
      matplotlib \
      gensim \
      wget \
      sklearn

RUN cd /tmp && \
    git clone https://github.com/$OC_ORIGIN/cogutil && \
    cd cogutil && \
    git checkout 60662c742be892f4ef77e40173f5b179e91b3926 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /tmp && \
    git clone https://github.com/$OC_ORIGIN/atomspace && \
    cd atomspace && \
    git checkout d439887bd380a1ae9b3c22e4fd146b54c6f709e2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /tmp && \
    git clone https://github.com/$OC_ORIGIN/ure && \
    cd ure && \
    git checkout cfc227b8ebccca716536fc89f954ef964b0d90a0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /tmp && \
    git clone https://github.com/$OC_ORIGIN/miner && \
    cd miner && \
    git checkout bdb6a773e958d41f119f7670809ac0868785be87 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /tmp && \
    wget https://github.com/leungmanhin/pln/archive/pln-morphism.zip && \
    unzip pln-morphism.zip && \
    rm pln-morphism.zip && \
    cd pln-pln-morphism && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /tmp && \
    git clone https://github.com/$OC_ORIGIN/agi-bio && \
    cd agi-bio && \
    git checkout 0cd9bd2a282032303e2666b89711a45b3bb6a086 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig

RUN rm -rf /tmp/*

RUN echo >> ~/.bashrc
RUN echo alias lt=\'ls -lth\' >> ~/.bashrc
RUN echo alias guileclean=\'rm -r $HOME/.cache/guile\' >> ~/.bashrc
RUN echo export GC_INITIAL_HEAP_SIZE=200G >> ~/.bashrc

RUN echo \(add-to-load-path \"/usr/share/guile/site/2.2/opencog\"\) >> ~/.guile
RUN echo \(add-to-load-path \".\"\) >> ~/.guile
RUN echo \(use-modules \(ice-9 readline\)\) >> ~/.guile
RUN echo \(activate-readline\) >> ~/.guile
RUN echo \(debug-enable \'backtrace\) >> ~/.guile
RUN echo \(read-enable \'positions\) >> ~/.guile
