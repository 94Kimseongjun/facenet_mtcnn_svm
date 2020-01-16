FROM continuumio/miniconda3 
MAINTAINER Seongjun Kim<s.kim@xiilab.com>

VOLUME /home/test
RUN conda update -n base conda
RUN yes | pip install tensorflow-gpu==1.15.0
RUN yes | pip install keras==2.3.1
RUN yes | pip install mtcnn
RUN yes | pip install matplotlib
RUN yes | pip install tqdm
RUN yes | pip install sklearn
RUN yes | pip install ipywidgets
RUN yes | pip install pillow
RUN yes | pip install pandas

WORKDIR /home






