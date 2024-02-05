# this fetches a pre-build base image of pytorch compiled for CUDA > 11.3,
# please use this as the base image for the RS server
FROM rsseminar/pytorch:latest

# Install dependencies
#RUN conda install -c conda-forge 'conda=4.10.3' cupy

# install dependencies
#RUN conda install -c conda-forge cupy  
RUN pip install opencv-python
RUN pip install scipy rasterio natsort matplotlib scikit-image tqdm natsort
RUN pip install s2cloudless
RUN pip install Pillow
RUN pip install dominate
RUN pip install visdom
RUN pip install wandb

# bake repository into dockerfile
RUN mkdir -p ./data
RUN mkdir -p ./scripts
RUN mkdir -p ./unet
RUN mkdir -p ./utils

ADD data ./data
ADD scripts ./scripts
ADD unet ./unet
ADD utils ./utils
ADD . ./

# this is setting your pwd at runtime
WORKDIR /workspace
