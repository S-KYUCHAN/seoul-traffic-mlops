import os
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './seoul-traffic-fd54b4ace57f.json'  # 로컬 키 경로

bucket_name = 'seoul-traffic-bucket'
train_blob  = 'training_dataset/train_20260308.parquet'  # 실제 파일명
test_blob   = 'training_dataset/test_20260308.parquet'

client = storage.Client()
bucket = client.bucket(bucket_name)

bucket.blob(train_blob).download_to_filename('./train.parquet')
bucket.blob(test_blob).download_to_filename('./test.parquet')
print("✅ 완료")