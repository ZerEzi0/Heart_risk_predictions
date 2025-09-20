FROM python:3.11-slim
# RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN apt-get update && apt-get install -y python3-opencv
COPY requirements.txt /app/requirements.txt
COPY src /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]