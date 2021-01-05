FROM ubuntu:18.04

# Use New Zealand mirrors
RUN sed -i 's/archive/nz.archive/' /etc/apt/sources.list

RUN apt update

# Set timezone to Auckland
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y locales tzdata
RUN locale-gen en_NZ.UTF-8
RUN dpkg-reconfigure locales
RUN echo "Pacific/Auckland" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata
ENV LANG en_NZ.UTF-8
ENV LANGUAGE en_NZ:en

# Install packages
RUN apt update && apt install -y silversearcher-ag entr tree

# Install python + other things
RUN apt install -y python3-dev python3-pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt

# Download transformer models
RUN python3 -c "import transformers;transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')"
RUN python3 -c "import transformers;transformers.BertForMaskedLM.from_pretrained('bert-base-cased')"
