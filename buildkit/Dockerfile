# syntax=docker/dockerfile:experimental

# Copyright (c) 2019,2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Please make sure you have nvidia-runtime enabled in docker config and build like:
#
# sudo -H DOCKER_BUILDKIT=1 nvidia-docker build -f Dockerfile.jetson -t nemo
#

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.2


# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PKG_CONFIG_PATH=..
ENV LLVM_CONFIG=/usr/bin/llvm-config-7
ENV MPLLOCALFREETYPE=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

# Parallel make by default
ENV MAKEOPTS="-j8"
#  Numpy
ENV NPY_NUM_BUILD_JOBS="8"
# Ninja
ENV MAX_JOBS="8"

# ccache
ENV CCACHE_DIR=/ccache
ENV CC="/usr/local/bin/gcc"
ENV CXX="/usr/local/bin/g++"

WORKDIR /tmp/nemo

COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN echo 'deb https://repo.download.nvidia.com/jetson/common r32.4 main' > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
RUN echo 'deb https://repo.download.nvidia.com/jetson/t194 r32.4 main' >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update

# Jetson CUDA Support
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y --no-install-recommends \
        libcublas-dev \
        cuda-nvcc-10-2 \
        cuda-cusparse-dev-10-2 \
        cuda-cudart-dev-10-2 \
        cuda-curand-dev-10-2 \
        cuda-cufft-dev-10-2 \
        cuda-tools-10-2 \
        cuda-cusolver-dev-10-2 \
        cuda-libraries-dev-10-2 \
        libcudnn8-dev \
        tensorrt \
        libnvinfer-dev \
        libnvonnxparsers-dev \
        libnvparsers-dev \
        python3-libnvinfer-dev

RUN ln -s /usr/local/cuda-10.2 /usr/local/cuda

# Everything else
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y \
libsndfile1 sox \
build-essential \
cmake \
pkg-config \
apt-utils \
libgoogle-glog0v5 \
libre2-dev \
libssl-dev \
libtool \
libboost-dev \
libopencv-dev \
python3-dev \
libffi-dev  \
llvm-7-dev \
llvm-9-dev \
liblzma-dev libbz2-dev \
rapidjson-dev \
python3-pyaudio \
libasound-dev \
portaudio19-dev \
libportaudio2 \
libportaudiocpp0 \
libsndfile1 \
alsa-base \
alsa-utils \
ffmpeg \
wget \
bc \
unzip \
parallel \
python3-distutils \
python3-pip \
libhdf5-serial-dev hdf5-tools libhdf5-dev \
libprotobuf-dev protobuf-compiler libopenblas-dev gfortran psmisc ccache libatlas-base-dev

# use ccache (make it appear in path earlier then /usr/bin/gcc etc)
RUN for p in gcc g++ cc c++; do ln -vs /usr/bin/ccache /usr/local/bin/$p;  done

#Install PyTorch 1.5
RUN wget -q https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl -O torch-1.5.0-cp36-cp36m-linux_aarch64.whl
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y libopenblas-base libopenmpi-dev
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install Cython
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install numpy torch-1.5.0-cp36-cp36m-linux_aarch64.whl

#Install torchvision  v0.6.0
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y git cmake libjpeg-dev zlib1g-dev
# RUN git clone --branch  v0.6.0 https://github.com/pytorch/vision torchvision
# RUN cd torchvision && pip3 install -e .
# RUN cd ../  && pip3 install 'pillow<7' 

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev
RUN git clone -b v0.1.85 https://github.com/google/sentencepiece/ sentencepiece-0.1.85
RUN --mount=type=cache,target=/ccache/ cd sentencepiece-0.1.85 && mkdir build && cd build && cmake .. && make && make install && ldconfig && \
    cd ../python && python3 setup.py bdist_wheel && cd dist && pip3 install sentencepiece-0.1.85-cp36-cp36m-linux_aarch64.whl --verbose

RUN --mount=type=cache,target=/ccache/ ccache -s

# Install latest 2019 tbb
RUN git clone -b tbb_2019 https://github.com/oneapi-src/oneTBB.git libtbb2
RUN --mount=type=cache,target=/ccache/ cd libtbb2 && \ 
    python build/build.py --prefix=/usr/local --install-libs --install-devel
RUN ldconfig

RUN --mount=type=cache,target=/ccache/ ccache -s

# Ninja
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install ninja

RUN --mount=type=cache,target=/ccache/ ccache -s

# Nvidia apex
RUN git clone https://github.com/NVIDIA/apex.git apex
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN ldconfig

RUN --mount=type=cache,target=/ccache/ ccache -s

# install nemo dependencies
WORKDIR /tmp/nemo

# Build the stuff that requires llvm-9 before it gets pulled as dep - do I need this anymore?
ENV LLVM_CONFIG=/usr/bin/llvm-config-9
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check scikit-learn
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check llvmlite

RUN --mount=type=cache,target=/ccache/ ccache -s

# Split these out to take advantage of docker build caching
ENV LLVM_CONFIG=/usr/bin/llvm-config-7
# Workaround
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check numba==0.48.0

# from docker_requirements.txt
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check boto3
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check frozendict
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check h5py
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check html2text
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check inflect
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check ipdb
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check ipython[all]
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check jupyterlab
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check kaldi-io
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check librosa
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check marshmallow
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y libpng-dev
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check matplotlib
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check nltk
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check num2words
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check onnx
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check pandas
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check pillow>=4.3.0
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check progressbar
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check requests
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check ruamel.yaml
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check sentencepiece
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check six>=1.14
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check sox
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check tqdm
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check unidecode
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check wget
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install --disable-pip-version-check youtokentome

RUN --mount=type=cache,target=/ccache/ ccache -s

# copy nemo source into a scratch image
WORKDIR /tmp/nemo-src
COPY . .

RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install -U pytest
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install megatron

RUN --mount=type=cache,target=/ccache/ ccache -s

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION=0.10.1
ARG BASE_IMAGE

# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "${NEMO_VERSION}" && \
/bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
/bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y curl portaudio19-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > r.sh
RUN chmod 777 ./r.sh && ./r.sh -y
ENV PATH=${HOME}/.cargo/bin:${PATH} 
RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip pip3 install setuptools_rust
#RUN PATH=${HOME}/.cargo/bin:${PATH} PYTHONPATH=${HOME}/.cargo/bin:${PYTHONPATH} pip3 install tokenizers

RUN --mount=type=cache,target=/ccache/ ccache -s

# RUN pip3 install torchaudio
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get install -y libsox-dev libsox-fmt-all
# This fails during docker build time. Install from running container to run examples/notebooks
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/targets/aarch64-linux/lib/"
# RUN mkdir -p /tmp/audio && cd /tmp/audio && git clone https://github.com/pytorch/audio.git -b v0.5.0 && cd audio && pip3 install .[all]

RUN --mount=type=cache,target=/ccache/ --mount=type=cache,target=/root/.cache/pip cd /tmp/nemo-src && \
    PATH=${HOME}/.cargo/bin:${PATH} PYTHONPATH=${HOME}/.cargo/bin:${PYTHONPATH} pip3 install ".[all]"

# copy scripts/examples/tests into container for end user
WORKDIR /workspace/nemo
COPY scripts /workspace/nemo/scripts
COPY examples /workspace/nemo/examples
COPY tests /workspace/nemo/tests
COPY README.rst LICENSE /workspace/nemo/

RUN --mount=type=cache,target=/ccache/ ccache -s

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
chmod +x start-jupyter.sh

