FROM tensorflow/tensorflow:latest-gpu

RUN pip install --upgrade pip
RUN pip install matplotlib pandas numpy pyyaml h5py scikit-learn opencv-python wandb dppy scipy seaborn

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install ffmpeg libsm6 libxext6
