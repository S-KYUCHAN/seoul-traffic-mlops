FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY train_mlp.py .

RUN pip install --no-cache-dir pandas pyarrow scikit-learn

ENTRYPOINT ["python", "train_mlp.py"]