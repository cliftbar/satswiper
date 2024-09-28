FROM python:3.11
LABEL authors="cliftbar"

WORKDIR satswiper

COPY ./requirements ./requirements


RUN sh ./requirements/install.sh
RUN mkdir tmp
RUN mkdir download

COPY ./src ./src
WORKDIR src

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
