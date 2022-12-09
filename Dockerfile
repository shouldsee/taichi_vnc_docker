

FROM gromacs/gromacs:2022.2


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && \
    apt-get install -y git \
      nano
RUN apt-get install -y moreutils
RUN apt-get install -y curl wget
RUN apt-get install -y build-essential gdb
RUN apt-get install -y python3.8 python3-pip python3-apt 
RUN apt-get install -y python3-socks
#RUN apt-get install -y software-properties-common gcc && \
#    add-apt-repository -y ppa:deadsnakes/ppa

# RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt


ARG GITURL

RUN python3 -m pip install -U pip
RUN python3 -m pip install https://github.com/shouldsee/pype/tarball/0.0.5
#RUN python3 -c "import pype; print(pype.__file__)" ; exit 1

RUN x=/usr/local/lib/python3.8/dist-packages/pype/controller.py ; grep -ve "typeguard" $x  | sponge $x

RUN python3 -m pip install supervisor
RUN python3 -m pip install flask==2.2.2
RUN python3 -m pip install apiflask==1.1.3

#ADD ./app/ /opt/app


ARG GITURL

#RUN mkdir -p /opt/app && cd /opt/app && \
#  git init . && \
#  git remote add origin $GITURL/mdsrv_app && \
#  git fetch origin --depth 1 21d0569009ff523b16ccb2acc1a0b4661fc36c61 && \
#  git reset --hard FETCH_HEAD


ADD BASHRC /tmp/BASHRC
RUN touch /root/.bashrc \
 && cat /tmp/BASHRC  >> /root/.bashrc

ARG APP_PORT
WORKDIR /opt
ENV APP_PORT=$APP_PORT

#python3 -m pip install flask==2.2.2
#CMD
#CMD cd /opt && python -m app.mdsrv --host 0.0.0.0 --port $APP_PORT --prefix /mdsrv 1>/data/stdout.log 2>/data/stderr.log

RUN apt-get install -y strace


CMD cd /data/; export FLASK_APP=/opt/app/server.py; \
  x=stderr-`date -Is`.log; touch $x; \
  ln -f $x ./ stderr-last.log; \
  python3 -m flask run -p 9002 -h 0.0.0.0 --reload  2>&1 | tee stderr-last.log