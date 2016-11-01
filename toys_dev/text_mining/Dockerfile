FROM continuumio/anaconda

USER root

# install basic linux / python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    liblapack-dev \
    libopenblas-dev \
    pkg-config \
    python-dev \
    python-protobuf \
    python-setuptools \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN easy_install pip

# install current dev version of scikit (for neural net library)
RUN pip install scikit-learn==0.18.rc2

# install beutiful soup - used to extract text from html
RUN pip install BeautifulSoup4

# install NLTK - used to stem / remove stop words
RUN pip install -U nltk
