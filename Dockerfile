FROM nvcr.io/nvidia/pytorch:21.06-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
#install specific versions of gdal
RUN apt install libgdal-dev=3.0.4+dfsg-1build3 -y
RUN pip install pygdal==3.0.4.6 albumentations
pip install timm==0.4.12 segmentation_models_pytorch cython tensorboardx madgrad
pip install opencv-python-headless==4.5.5.64
apt install -y libgl1
