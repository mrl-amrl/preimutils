FROM jjanzic/docker-python3-opencv:latest

COPY ./preimutils /app/preimutils
COPY ./requirement.txt /
RUN pip install -r requirement.txt && rm -fr /root/.cache
WORKDIR /app