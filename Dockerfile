FROM wuhanchu/centos:torch

WORKDIR /opt/service/
COPY "./" "./"

RUN chmod +x ./run.sh && python3 init_model.py && \
    pip3 install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir

CMD ./run.sh
