FROM python:3.6.9-slim-buster

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 curl build-essential

RUN python3 -m pip install torch torchvision opencv-python

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY run_inference.py /opt/run_inference.py
COPY sample_frame.jpg /opt/sample_frame.jpg

WORKDIR /opt

ENTRYPOINT [ "python3", "run_inference.py" ]
