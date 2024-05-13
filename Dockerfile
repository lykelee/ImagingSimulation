FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && apt-get update

RUN apt-get -y install git libgl1-mesa-glx libglib2.0-0 vim

RUN pip install \
cached-property==1.5.2 \
certifi==2021.10.8 \
h5py==3.6.0 \
matplotlib \
natsort \
numpy==1.21.5 \
opencv-python==4.5.5.62 \
Pillow==6.2.2 \
scikit-image \
scipy==1.7.3 \
six==1.16.0 \
tensorboardX \
tifffile==2021.11.2 \
torch==1.2.0 \
torchvision==0.4.0

#CMD echo test