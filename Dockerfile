FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


RUN apt-get update
RUN apt-get install -y --no-install-recommends wget git vim software-properties-common 
RUN apt-get install -y --no-install-recommends g++-5 gcc-5

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
RUN update-alternatives --set cc /usr/bin/gcc
RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
RUN update-alternatives --set c++ /usr/bin/g++

RUN apt-get install -y --no-install-recommends libboost-all-dev

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev libgl1-mesa-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN echo 'alias python="python3.6"' >> /root/.bashrc
RUN echo 'alias pip="pip3"' >> /root/.bashrc
#RUN echo 'export CUDA_HOME=/usr/local/cuda' >> /root/.bashrc
#RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64' >> /root/.bashrc

WORKDIR /root
RUN git clone https://github.com/shuto-keio/3DSSD-TFv2.git

WORKDIR 3DSSD-TFv2

RUN python -V
RUN pip install -r requirements.txt
RUN pip install tensorflow-gpu==2.3.0
RUN ln -s /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.2 /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so

ENV PYTHONPATH $PYTHONPATH:/root/3DSSD-TFv2/lib:/root/3DSSD-TFv2/myavi
RUN bash compile_all.sh /usr/local/lib/python3.6/dist-packages/tensorflow /usr/local/cuda

WORKDIR /root
