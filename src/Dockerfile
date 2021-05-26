FROM python:3.8
LABEL student1="Wojciech Lulek" student2="Rafał Ewiak"

#COPY ./src ./app/src

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install opencv-python numpy

WORKDIR "./app/src"

ENTRYPOINT ["python","/app/src/main.py"]