FROM python:3.8
LABEL student1="Wojciech Lulek" student2="Rafał Ewiak"

COPY ./src ./app
COPY ./requirements.txt ./app/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -r ./app/requirements.txt

WORKDIR "./app/"

ENV PYTHONPATH "${PYTHONPATH}:/app/"

ENTRYPOINT ["python", "./main.py"]