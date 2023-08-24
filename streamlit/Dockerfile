FROM python:3.8.15-slim-buster

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]