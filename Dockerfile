FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Thiết lập môi trường không cần tương tác (tránh hỏi YES/NO khi cài)
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /dl

COPY voc_dataset.py ./voc_dataset.py
COPY train_faster_rcnn.py ./train.py

RUN apt-get update
RUN apt-get install -y vim

RUN pip install tensorboard torchmetrics
RUN pip install torchmetrics[detection]
