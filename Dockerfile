# FROM python:3.8.6-buster

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

ENV GDRIVE_LINK https://drive.google.com/drive/folders/18qRI1ZJMBiGXa_8GArn3_68DaA4BNfF7?usp=sharing

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

RUN mkdir models_data && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    git clone https://github.com/matthuisman/gdrivedl.git && \
    python gdrivedl/gdrivedl.py $GDRIVE_LINK -P models_data

WORKDIR $PROJECT_ROOT

RUN pip install -r requirements.txt

CMD python lib/run.py
