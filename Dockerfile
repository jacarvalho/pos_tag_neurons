FROM ubuntu:16.04
MAINTAINER Joao Carvalho <carvalhj@cs.uni-freiburg.de>

# Set up basic tools
RUN apt-get update && apt-get install -y make vim
RUN apt-get install -y python3 python3-pip python3-dev build-essential
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade virtualenv
RUN pip3 install --upgrade setuptools pip

# Copy necessary files to /code
RUN mkdir -p /code
COPY . /code/

# Install python libraries
RUN pip3 install -r /code/requirements.txt
RUN python3 -m nltk.downloader treebank
RUN python3 -m nltk.downloader maxent_treebank_pos_tagger
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader averaged_perceptron_tagger

# cd to /code/www/ directory
WORKDIR /code

# Start the webapp - NOTE: a dockerfile can have only one CMD
# CMD ["python3", "app.py"]


# docker build -t code .
# docker run -it -p 5000:5000 -v /nfs/students/joao-carvalho/:/extern/data code
