ARG PYTORCH="2.7.1"
ARG CUDA="12.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX" \
    && export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    && export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" 

# INIT OF FIX FOR GPG KEY
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Add the current user to the image
ARG USER_NAME=vmr_user
ARG USER_ID=1000
ARG GROUP_NAME=vmr_group
ARG GROUP_ID=1000

RUN groupadd --gid $GROUP_ID $GROUP_NAME \
    && useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && usermod -aG sudo $USER_NAME

# ---------------------------
# Install Miniconda
# ---------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# ---------------------------
# Copy environment files from host
# ---------------------------
COPY environment_cgdetr.yml /project/environment.yml

# Ensure latest conda and pip
RUN conda update -n base -c defaults conda -y \
    && pip install --upgrade pip setuptools wheel

# Install key dependencies from pip directly
RUN pip install \
    attrs==23.2.0 \
    cmake==3.26.3 \
    datasets==3.2.0 \
    decorator==4.4.2 \
    einops==0.8.0 \
    ffmpeg-python==0.2.0 \
    huggingface-hub==0.32.4 \
    imageio==2.33.1 \
    jsonlines==4.0.0 \
    markdown==3.6 \
    matplotlib==3.7.1 \
    moviepy==1.0.3 \
    multiprocess==0.70.16 \
    ninja==1.11.1.4 \
    nltk==3.9.1 \
    numpy==1.26.4 \
    opencv-python==4.9.0.80 \
    pandas \
    scipy \
    tensorboard \
    seaborn \
    scikit-learn \
    tqdm \
    yacs \
    tabulate 

USER $USER_NAME
ENV PROJECT_PATH=/project
WORKDIR $PROJECT_PATH

# CUDA / PyTorch flags (optional)
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda

CMD ["/bin/bash"]
