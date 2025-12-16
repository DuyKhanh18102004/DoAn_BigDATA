FROM apache/spark-py:v3.3.0

USER root

# install system deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    tensorflow==2.11.0 \
    Pillow==9.5.0 \
    numpy==1.23.5 \
    keras==2.11.0 \
    h5py==3.8.0

WORKDIR /app
