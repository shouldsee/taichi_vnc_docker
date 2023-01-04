
#FROM gromacs/gromacs:2022.2
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM dorowu/ubuntu-desktop-lxde-vnc


RUN  apt-key adv --fetch-keys https://dl.google.com/linux/linux_signing_key.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && \
    apt-get install -y git \
      nano
RUN apt-get install -y moreutils
RUN apt-get install -y curl wget
RUN apt-get install -y build-essential gdb

RUN apt-get install -y python3.8 python3-pip python3-apt
RUN apt-get install -y python3-socks


RUN  apt-key adv --fetch-keys http://packages.lunarg.com/lunarg-signing-key-pub.asc
RUN \
  curl -L -o /etc/apt/sources.list.d/lunarg-vulkan-focal.list \
  http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
  
# RUN cat /etc/apt/sources.list.d/lunarg-vulkan-focal.list  && exit 1

RUN apt update 
RUN apt-get install -y vulkan-sdk
RUN apt-get install -y mesa-vulkan-drivers

ARG GITURL

RUN python3 -m pip install -U pip

RUN python3 -m pip install taichi==1.3.0

ARG GITURL



COPY BASHRC /tmp/BASHRC
RUN touch /root/.bashrc \
  && cat /tmp/BASHRC  >> /root/.bashrc

ARG APP_PORT
WORKDIR /opt
ENV APP_PORT=$APP_PORT


ENV SERVICE_NAME="gcat_gromacs"
ENV SERVICE_IP="192.168.50.132"
ENV SERVICE_PORT=$APP_PORT


CMD bash
