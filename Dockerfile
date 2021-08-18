FROM centos:centos7

WORKDIR /opt/service/
COPY "./" "./"

RUN chmod +x ./run.sh

RUN echo 'export LC_ALL="en_US.UTF-8"' >> /etc/profile && echo 'export LC_ALL="en_US.UTF-8"' >> /etc/profile && source /etc/profile
RUN yum -y install python3 wget
RUN pip3 install ./wheels/sentencepiece-0.1.96-cp36-cp36m-linux_2_17_x86_64.linux_x86_64.whl
RUN pip3 install ./wheels/torch-1.9.0-cp36-cp36m-manylinux1_x86_64.whl
RUN pip3 install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ./run.sh
