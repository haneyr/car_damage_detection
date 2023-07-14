FROM nvcr.io/nvidia/pytorch:23.06-py3

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update && apt-get install -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

CMD exec gunicorn --bind :$AIP_HTTP_PORT --workers 4 --threads 8 --timeout 0 main:app
