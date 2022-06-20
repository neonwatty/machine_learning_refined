FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get update
RUN apt-get install -y python3.8
RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get update
RUN apt-get install -y python3.8-dev 
RUN apt-get update
RUN apt-get install -y default-libmysqlclient-dev build-essential
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libfreetype-dev libfreetype6 libfreetype6-dev 
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
RUN apt-get update
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt