# -*- mode: sh -*-
exec >&2

TMP=$(mktemp)
cat > "$TMP" << EOF
Bootstrap: docker
From: nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

%environment
    # only for build process
    export DEBIAN_FRONTEND=noninteractive
    export PIP_ROOT_USER_ACTION=ignore

    # generally necessary
    export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
    export TZ=Europe/Berlin
    export DGLBACKEND=pytorch
    # export SHELL /bin/bash

%post
    apt-get -q update

    apt-get install -y  -qq --reinstall software-properties-common && \
            apt-get -qq update && apt-get upgrade -y -qq && \
            apt-get install -yq -qq --no-install-recommends \
            build-essential \
            ca-certificates \
            cmake \
            curl \
            doxygen \
            gfortran \
            git \
            libavcodec-dev \
            libavformat-dev \
            libavutil-dev \
            libblas-dev \
            libboost-all-dev \
            libcurl3-dev \
            libeigen3-dev \
            libfftw3-dev \
            libflann-dev \
            libfreetype6-dev \
            libglew-dev \
            libgoogle-perftools-dev \
            libgsl-dev \
            libgtk2.0-dev \
            libjpeg-dev \
            libjs-mathjax \
            liblapack-dev \
            liblz4-dev \
            libmetis-dev \
            libpng-dev \
            libpostproc-dev \
            libpq-dev \
            libprotobuf-dev \
            libswscale-dev \
            libtbb-dev \
            libtiff-dev \
            libtiff5-dev \
            libturbojpeg0-dev \
            libxine2-dev \
            libzmq3-dev \
            meson \
            ninja \
            pkg-config \
            poppler-utils \
            pwgen \
            python3 \
            python3-dev \
            python3-pip \
            python3-venv \
            rsync \
            software-properties-common \
            sudo \
            swig \
            tmux \
            unzip \
            vim \
            wget \
            yasm \
            zip \
            zlib1g-dev \
            && apt-get clean -qq \
            && rm -rf /var/lib/apt/lists/*

     pip install --break-system-packages \
         torch==2.4.0 \
          torchvision==0.19.0 \
           --index-url https://download.pytorch.org/whl/cu124

     pip install --upgrade --break-system-packages \
            "networkx>=3.4.2" \
            "pandas[performance,parquet]>=2.2.3" \
            "numpy>=2.0.0" \
            "scikit-learn>=1.6.0" \
            "annoy>=1.17.3" \
            "matplotlib>=3.9.3" \
            "opentsne>=1.0.2" \
            "tsimcne>=0.4.13" \
            "lightning>=2.4.0" \
            "scipy>=1.14.1" \
            "numba>=0.60.0" \
            cupy-cuda12x \
            "opentsne>=1.0.2" \
            ipython \
            black \
            sphinx \
            meson \
            ninja \
            "polars>=1.17.1" \
            "lightning>=2.4.0" \
            "python-telegram-bot>=21.9" \
            humanize \
            "ogb>=1.3.6" \
            "torch_geometric>=2.6.1" \
            "contrastive-ne>=0.3.8" \
            git+https://github.com/jnboehm/t-fdp \

    pip install --break-system-packages \
        pyg-lib torch_scatter torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
        && \
        pip install --break-system-packages dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
EOF

# on my laptop, I need to run:
# run0 --setenv=APPTAINER_TMPDIR=$PWD apptainer build nik.sif nik.def

# set default values, if not set
: ${XDG_CACHE_DIR:=$HOME/.cache}
env SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR-$XDG_CACHE_DIR/singularity} \
    SINGULARITY_TMPDIR=${SCRATCH-$PWD} \
        singularity build --fakeroot $3 $TMP
rm $TMP
