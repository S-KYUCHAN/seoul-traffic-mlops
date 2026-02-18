FROM python:3.10.0-slim 
WORKDIR /app 
COPY requirements.txt /app/requirements.txt 
COPY tune.py /app/tune.py 
RUN pip install -r requirements.txt 
RUN rm requirements.txt 
CMD ["python", "/app/tune.py"]
