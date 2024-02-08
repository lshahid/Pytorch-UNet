# FROM nvcr.io/nvidia/pytorch:22.11-py3
FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/autoseg_2D

ADD requirements.txt .
RUN pip3 install --no-cache-dir --upgrade --pre pip
RUN pip3 install --no-cache-dir -r requirements.txt
ADD . .

WORKDIR /workspace
