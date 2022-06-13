FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get update
RUN apt-get install -y python3.8
RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get update
RUN apt-get install -y python-dev default-libmysqlclient-dev

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt