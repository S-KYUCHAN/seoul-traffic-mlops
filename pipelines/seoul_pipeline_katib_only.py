import kfp
from kfp import kubernetes
from kfp import dsl
from kfp.dsl import Output, Input, Dataset

SEOUL_BASE_IMAGE="mlops.kr-central-2.kcr.dev/kc-kubeflow-registry/jupyter-scipy:v1.10.0.py311.1a"

@dsl.component(base_image=SEOUL_BASE_IMAGE)
def featurization(
    source_csv: str,
    data_mnt_path: str,
    start_date: str,
    eval_date: str,
):
    import pandas as pd
    import numpy as np
    import os

    file_path = os.path.join(data_mnt_path, source_csv)
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    
    hour_sr = df['datetime'].dt.hour
    dow_sr = df['datetime'].dt.dayofweek
    
    dataset = pd.DataFrame()
    dataset['datetime'] = df['datetime']  # 이후 작업 편의를 위해 설정
    
    # 시간 관련 feature
    dataset['x1'] = np.sin(2*np.pi*hour_sr/24)
    dataset['x2'] = np.cos(2*np.pi*hour_sr/24)
    # 요일 관련 feature
    dataset['x3'] = np.sin(2*np.pi*dow_sr/7)
    dataset['x4'] = np.cos(2*np.pi*dow_sr/7)
    
    # 예측하고자 하는 대상, label
    dataset['y'] = df['vol']

    eval_dt = pd.to_datetime(eval_date)
    train_df = dataset[dataset['datetime'] < eval_dt]
    test_df = dataset[dataset['datetime'] >= eval_dt]

    train_df.to_csv(os.path.join(data_mnt_path,f'seoul-{start_date}.csv'), index=False)
    test_df.to_csv(os.path.join(data_mnt_path,f'seoul-{start_date}-test.csv'), index=False)

@dsl.component(base_image=SEOUL_BASE_IMAGE, packages_to_install=['kubeflow-katib', 'scikit-learn==1.6.1'])
def tuning_with_katib(
    data_mnt_path: str,
    model_mnt_path: str,
    artifact_mnt_path: str,
    katib_exp_name: str,
    model_version: str,
    start_date: str,
    exp_suffix: str,
):
    import os
    import ast
    import joblib,json
    import re, hashlib

    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    
    import kubeflow.katib as katib

    def mk_katib_name(base: str, start_date: str, suffix: str) -> str:
        s = f"{base}-d{start_date}-{suffix}".lower()
        s = re.sub(r'[^a-z0-9-]+', '-', s).strip('-')
        if not re.match(r'^[a-z]', s):
            s = "k-" + s
        if len(s) > 40:
            h = hashlib.sha1(s.encode()).hexdigest()[:6]
            s = s[:33].rstrip('-') + "-" + h
        return s
    
    katib_client = katib.KatibClient()
    
    katib_exp = katib_client.get_experiment(katib_exp_name)
    katib_exp.metadata.name = mk_katib_name(katib_exp_name, start_date, exp_suffix)
    exp_name = katib_exp.metadata.name
    katib_exp.metadata.resource_version = None
    
    container_spec = katib_exp.spec.trial_template.trial_spec['spec']['template']['spec']['containers'][0]
    container_spec.setdefault('args', [])
    container_spec['args'].extend(['--start_date', start_date])
    
    katib_client.create_experiment(katib_exp)
    katib_client.wait_for_experiment_condition(name=exp_name)

    
    optim_params = katib_client.get_optimal_hyperparameters(exp_name)
    print(optim_params)
    
    params = { param.name: ast.literal_eval(param.value)   for param in optim_params.parameter_assignments}


    df = pd.read_csv(os.path.join(data_mnt_path,f'seoul-{start_date}.csv'))
    print("read df")
    X = df[['x1', 'x2', 'x3', 'x4']]
    y = df['y']
    
    
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    print("Model train")

    
    model_dir = os.path.join(model_mnt_path, model_version)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir,'model.joblib'))
    print("Dump to model")
    
    artifact_dir = os.path.join(artifact_mnt_path, model_version)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, 'param.json'), 'w') as f:
        json.dump(params,f)

@dsl.component(
    base_image=SEOUL_BASE_IMAGE,
    packages_to_install=["scikit-learn==1.6.1", "joblib"],
)
def evaluate_model(
    data_mnt_path: str,
    model_mnt_path: str,
    artifact_mnt_path: str,
    model_version: str,
    start_date: str,
):
    import os
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    df = pd.read_csv(os.path.join(data_mnt_path, f"seoul-{start_date}-test.csv"))
    X = df[["x1", "x2", "x3", "x4"]]
    real = df["y"]

    model_path = os.path.join(model_mnt_path, model_version, "model.joblib")
    model = joblib.load(model_path)

    pred = model.predict(X)
    mae = mean_absolute_error(real, pred)

    metrics = {"mae": float(mae)}
    print(f"MAE: {mae}")

    artifacts_path = os.path.join(artifact_mnt_path, model_version)
    os.makedirs(artifacts_path, exist_ok=True)
    with open(os.path.join(artifacts_path, "score.json"), "w") as f:
        json.dump(metrics, f)
    print("Saved metrics to:", os.path.join(artifacts_path, "score.json"))

    with open(os.path.join(artifacts_path, "model_version.txt"), "w") as f:
        f.write(model_version)
    print("Saved model version to :", os.path.join(artifacts_path, "model_version.txt"))

@dsl.pipeline(name="Load Prediction Model Pipeline (Katib only)")
def seoul_model_pipeline_katib_only(
    source_data: str,
    start_date: str,
    eval_date: str,
    namespace: str,  # 유지: run에서 입력하던 값 그대로 받되, 지금은 내부 로직에선 안 써도 됨
    dataset_pvc: str = "dataset-pvc",
    model_pvc: str = "model-pvc",
    artifact_pvc: str = "artifact-pvc",
    katib_template_exp_name: str = "seoul-gbr-tune",
    model_name: str = "seoul-traffic-predictor", 
):
    # Katib experiment 충돌 방지 suffix
    exp_suffix = dsl.PIPELINE_JOB_NAME_PLACEHOLDER

    # # step1: Load dataset
    # load_dataset_task = preprocessing(
    #     source_data=source_data,
    #     data_mnt_path="/dataset",
    #     start_date=start_date,
    #     eval_date=eval_date,
    # )
    # kubernetes.mount_pvc(load_dataset_task, pvc_name=dataset_pvc, mount_path="/dataset")

    # step2: Featurization
    featurization_task = featurization(
        source_csv=source_data,
        data_mnt_path="/dataset",
        start_date=start_date,
        eval_date=eval_date,
    )
    kubernetes.mount_pvc(featurization_task, pvc_name=dataset_pvc, mount_path="/dataset")

    # step3: Katib Tuning + Train final model
    tuning_model_task = tuning_with_katib(
        data_mnt_path="/dataset",
        model_mnt_path="/model",
        artifact_mnt_path="/artifact",
        katib_exp_name=katib_template_exp_name,
        model_version=start_date,
        start_date=start_date,
        exp_suffix=exp_suffix,
    ).after(featurization_task)

    kubernetes.mount_pvc(tuning_model_task, pvc_name=dataset_pvc, mount_path="/dataset")
    kubernetes.mount_pvc(tuning_model_task, pvc_name=model_pvc, mount_path="/model")
    kubernetes.mount_pvc(tuning_model_task, pvc_name=artifact_pvc, mount_path="/artifact")

    # step4: Evaluation
    evaluate_model_task = evaluate_model(
        data_mnt_path="/dataset",
        model_mnt_path="/model",
        artifact_mnt_path="/artifact",
        model_version=start_date,
        start_date=start_date,
    ).after(tuning_model_task)

    kubernetes.mount_pvc(evaluate_model_task, pvc_name=dataset_pvc, mount_path="/dataset")
    kubernetes.mount_pvc(evaluate_model_task, pvc_name=model_pvc, mount_path="/model")
    kubernetes.mount_pvc(evaluate_model_task, pvc_name=artifact_pvc, mount_path="/artifact")


# 컴파일
kfp.compiler.Compiler().compile(seoul_model_pipeline_katib_only, "seoul-pipeline-katib-only.yaml")
print("✅ compiled: seoul-pipeline-katib-only.yaml")
