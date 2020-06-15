FROM python:2.7
LABEL maintainer javiteri

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p input/orig/
RUN mkdir -p input/pre/
RUN mkdir output/
COPY models/unet2/_85_15_unet2.h5 the_model.h5
COPY test_and_create_one.py test_and_create_one.py
ENTRYPOINT ["bash"]