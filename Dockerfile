# Joao Carvalho <carvalhj@cs.uni-freiburg.de>
# Copyright 2018

# This Dockerfile builds a docker image and runs a webapp to showcase the
# results of the master project

FROM ubuntu:16.04
MAINTAINER Joao Carvalho <carvalhj@cs.uni-freiburg.de>

# Set up basic tools
RUN apt-get update && apt-get install -y make vim
RUN apt-get install -y python3 python3-pip python3-dev build-essential
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade virtualenv
RUN pip3 install --upgrade setuptools pip

# Copy necessary files to /
COPY . /

# cd to / directory
WORKDIR /

# Install python libraries and packages
RUN pip3 install -r requirements.txt
RUN python3 setup.py install
 
# Start the webapp
# NOTE: a dockerfile can have only one CMD
WORKDIR /www
CMD ["python3", "/www/app.py"]


# docker build -t project .
# docker run -it -p 5000:5000 -v /abs/path/to/repository/:/extern/data project
