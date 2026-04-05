FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# serving에 필요한 패키지만 추가 설치
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pandas \
    mlflow==3.7.0
    # mlflow는 꼭 필요하진 않지만, 모델 저장 환경과 최대한 맞추고 싶으면 유지

WORKDIR /app

COPY app.py .

# 모델이 /model/v1 아래 있다고 가정 (PVC 마운트)
ENV MODEL_DIR=/model/v1

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]