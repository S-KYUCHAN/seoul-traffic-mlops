import pandas as pd
import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Output, Input, Dataset

SEOUL_BASE_IMAGE="mlops.kr-central-2.kcr.dev/kc-kubeflow-registry/jupyter-scipy:v1.10.0.py311.1a"
MLFLOW_BASE_IMAGE = "seoul-traffic-mlflow:latest"
NAMESPACE = "kubeflow-user-example-com"

@dsl.component(
    base_image=SEOUL_BASE_IMAGE, 
    packages_to_install=['google-cloud-storage']
    )
def fetch_from_gcs(
    bucket_name: str,
    train_blob: str,
    test_blob: str,
    data_mnt_path: str,
):
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/secret/gcp/key.json'

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    train_path = os.path.join(data_mnt_path, 'train.parquet')
    test_path = os.path.join(data_mnt_path, 'test.parquet')

    bucket.blob(train_blob).download_to_filename(train_path)
    print(f"fetched train data: {train_blob} -> {train_path}")

    bucket.blob(test_blob).download_to_filename(test_path)
    print(f"fetched test data: {test_blob} -> {test_path}")

@dsl.component(base_image=SEOUL_BASE_IMAGE)
def featurization(
    data_mnt_path: str,
    train_file: str,
    test_file: str,
    model_version: str,
):
    import numpy as np
    import pandas as pd
    import os

    train_df = pd.read_parquet(os.path.join(data_mnt_path, train_file))
    test_df = pd.read_parquet(os.path.join(data_mnt_path, test_file))

    def featurize(df):
        df = df[df['quality_flag'] == 'Y'].copy()
        df['datetime'] = pd.to_datetime(df['ymd']) + pd.to_timedelta(df['hh'], unit='h')
        hour_sr = df['datetime'].dt.hour
        dow_sr = df['datetime'].dt.dayofweek
        # 시간 관련 feature
        df['x1'] = np.sin(2*np.pi*hour_sr/24)
        df['x2'] = np.cos(2*np.pi*hour_sr/24)
        # 요일 관련 feature
        df['x3'] = np.sin(2*np.pi*dow_sr/7)
        df['x4'] = np.cos(2*np.pi*dow_sr/7)
        df['y'] = df['total_vol']
        return df[['x1', 'x2', 'x3', 'x4', 'y']]

    featurize(train_df).to_csv(os.path.join(data_mnt_path, f'train-{model_version}.csv'), index=False)
    featurize(test_df).to_csv(os.path.join(data_mnt_path, f'test-{model_version}.csv'), index=False)
    print(f"Featurization done: {model_version}")

@dsl.component(
    base_image=SEOUL_BASE_IMAGE,
    packages_to_install=['kubeflow-katib']
)
def tuning_with_katib(
    katib_exp_name: str,
    model_version: str,
    exp_suffix: str,
    namespace: str = "kubeflow-user-example-com",
) -> str:
    import kubeflow.katib as katib
    import re, hashlib, json
    import ast

    def make_katib_name(base: str, version: str, suffix: str) -> str:
        s = f"{base}-d{version}-{suffix}".lower()
        s = re.sub(r'[^a-z0-9-]+', '-', s).strip('-')
        if not re.match(r'^[a-z]', s):
            s = "k-" + s
        if len(s) > 40:
            h = hashlib.sha1(s.encode()).hexdigest()[:6]
            s = s[:33].rstrip('-') + "-" + h
        return s

    katib_client = katib.KatibClient(namespace=namespace)
    katib_exp = katib_client.get_experiment(katib_exp_name)
    exp_name = make_katib_name(katib_exp_name, model_version, exp_suffix)
    katib_exp.metadata.name = exp_name
    katib_exp.metadata.resource_version = None

    katib_client.create_experiment(katib_exp)
    katib_client.wait_for_experiment_condition(name=exp_name)

    optim_params = katib_client.get_optimal_hyperparameters(exp_name)
    print(f"optim_params: {optim_params}")

    params = { param.name: ast.literal_eval(param.value) for param in optim_params.parameter_assignments}

    print(f"Best params: {params}")

    return json.dumps(params)

@dsl.component(
    base_image=MLFLOW_BASE_IMAGE,
)
def train_final_model(
    best_params_json: str,
    data_mnt_path: str,
    model_mnt_path: str,
    artifact_mnt_path: str,
    model_version: str,
):
    import os, json
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import mlflow
    import mlflow.pytorch
    
    MLFLOW_URI = "http://mlflow.kubeflow-user-example-com.svc.cluster.local:80"
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("seoul-traffic-mlp")

    params = json.loads(best_params_json)
    print(f"params: {params}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class TrafficDataset(Dataset):
        def __init__(self, csv_path):
            df = pd.read_csv(csv_path)

            self.Y_mean = df['y'].mean()
            self.Y_std = df['y'].std()

            df['y_norm'] = (df['y'] - self.Y_mean) / self.Y_std

            self.X = torch.tensor(df[['x1', 'x2', 'x3', 'x4']].values, dtype=torch.float32)
            self.Y = torch.tensor(df['y_norm'].values, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    class TrafficMLP(nn.Module):
        def __init__(self, hidden_size=128, num_layers=3):
            super().__init__()
            layers = [nn.Linear(4, hidden_size), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            layers.append(nn.Linear(hidden_size, 1))
            self.net = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.net(x)

    data_path = os.path.join(data_mnt_path, f"train-{model_version}.csv")
    dataset = TrafficDataset(data_path)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = TrafficMLP(
        hidden_size=params["hidden_size"], 
        num_layers=params["num_layers"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    with mlflow.start_run(run_name = model_version):
        mlflow.log_params(params)

        for epoch in range(params["num_epochs"]):
            model.train()
            total_loss = 0.0

            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        pred = model(X_batch).squeeze(-1)
                        loss = loss_fn(pred, Y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch).squeeze(-1)
                    loss = loss_fn(pred, Y_batch)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    all_X = dataset.X.to(DEVICE)
                    all_Y = dataset.Y.to(DEVICE)

                    if scaler:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            pred_norm = model(all_X).squeeze(-1)
                    else:
                        pred_norm = model(all_X).squeeze(-1)
                    pred_vol = pred_norm * dataset.Y_std + dataset.Y_mean
                    real_vol = all_Y * dataset.Y_std + dataset.Y_mean
                    mae = (pred_vol - real_vol).abs().mean().item()

                mlflow.log_metric("train_mae", mae, step=epoch+1)
                print(f"Epoch {epoch+1}/{params['num_epochs']} | Loss: {total_loss/len(loader):.4f} | MAE: {mae:.1f}")

        model_dir = os.path.join(model_mnt_path, model_version)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))

        mlflow.pytorch.log_model(model, artifact_path="model")

        artifact_dir = os.path.join(artifact_mnt_path, model_version)
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, 'param.json'), 'w') as f:
            json.dump({
                **params,
                "Y_mean": dataset.Y_mean,
                "Y_std": dataset.Y_std,
            }, f)
        print(f"Saved params to {artifact_dir}/param.json")

@dsl.component(
    base_image= MLFLOW_BASE_IMAGE,
)
def evaluate_model(
    data_mnt_path: str,
    model_mnt_path: str,
    artifact_mnt_path: str,
    model_version: str,
):
    import os, json
    import torch
    import torch.nn as nn
    import pandas as pd
    import mlflow
    
    MLFLOW_URI = "http://mlflow.kubeflow-user-example-com.svc.cluster.local:80"
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("seoul-traffic-mlp")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    class TrafficMLP(nn.Module):
        def __init__(self, hidden_size=128, num_layers=3):
            super().__init__()
            layers = [nn.Linear(4, hidden_size), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            layers.append(nn.Linear(hidden_size, 1))
            self.net = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.net(x)

    with open(os.path.join(artifact_mnt_path, model_version, "param.json")) as f:
        params = json.load(f)
        
    model = TrafficMLP(
        hidden_size=params["hidden_size"], 
        num_layers=params["num_layers"]
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(
            os.path.join(model_mnt_path, model_version, "model.pt"),
            map_location=DEVICE
        )
    )
    model.eval()

    df = pd.read_csv(os.path.join(data_mnt_path, f"test-{model_version}.csv"))
    X = torch.tensor(df[['x1', 'x2', 'x3', 'x4']].values, dtype=torch.float32).to(DEVICE)
    Y_mean = params["Y_mean"]
    Y_std = params["Y_std"]
    y_real = df['y'].values

    with torch.no_grad():
        pred_norm = model(X).squeeze(-1).cpu().numpy()
    pred_vol = pred_norm * Y_std + Y_mean

    mae = float(abs(pred_vol - y_real).mean())
    print(f"MAE: {mae:.2f}")

    with mlflow.start_run(run_name=f"{model_version}-eval"):
        mlflow.log_metric("mae", mae)
        print(f"MLflow logged MAE: {mae}")

    artifacts_path = os.path.join(artifact_mnt_path, model_version)
    os.makedirs(artifacts_path, exist_ok=True)
    with open(os.path.join(artifacts_path, "score.json"), "w") as f:
        json.dump({"mae": mae}, f)

    print(f"Saved score to {artifacts_path}/score.json")


@dsl.pipeline(name="seoul-traffic-mlp-pipeline")
def pipeline(
    bucket_name: str = "seoul-traffic-bucket",
    train_blob: str = "training_dataset/train_20260308.parquet",
    test_blob: str = "training_dataset/test_20260308.parquet",
    model_version: str = "v1",
    data_mnt_path: str = "/data",
    model_mnt_path: str = "/model",
    artifact_mnt_path: str = "/artifact",
    katib_exp_name: str = "seoul-traffic-mlp-tuning",
    namespace: str = NAMESPACE,
):
    exp_suffix = dsl.PIPELINE_JOB_NAME_PLACEHOLDER

    fetch_task = fetch_from_gcs(
        bucket_name = bucket_name,
        train_blob= train_blob,
        test_blob = test_blob,
        data_mnt_path = data_mnt_path,
    )
    kubernetes.mount_pvc(fetch_task, pvc_name="data-pvc", mount_path="/data")
    kubernetes.use_secret_as_volume(
        fetch_task,
        secret_name="gcp-sa-key",
        mount_path="/secret/gcp",
    )

    featurize_task = featurization(
        data_mnt_path = data_mnt_path,
        train_file = "train.parquet",
        test_file = "test.parquet",
        model_version = model_version,
    ).after(fetch_task)
    kubernetes.mount_pvc(featurize_task, pvc_name="data-pvc", mount_path="/data")

    katib_task = tuning_with_katib(
        katib_exp_name = katib_exp_name,
        model_version = model_version,
        exp_suffix = exp_suffix,
        namespace = NAMESPACE,
    ).after(featurize_task)

    train_task = train_final_model(
        best_params_json = katib_task.output,
        data_mnt_path = data_mnt_path,
        model_mnt_path = model_mnt_path,
        artifact_mnt_path = artifact_mnt_path,
        model_version = model_version,
    ).after(katib_task)
    kubernetes.mount_pvc(train_task, pvc_name="data-pvc", mount_path="/data")
    kubernetes.mount_pvc(train_task, pvc_name="model-pvc", mount_path="/model")
    kubernetes.mount_pvc(train_task, pvc_name="artifacts-pvc", mount_path="/artifact")
    kubernetes.set_image_pull_policy(train_task, "Never")

    evaluate_task = evaluate_model(
        data_mnt_path = data_mnt_path,
        model_mnt_path = model_mnt_path,
        artifact_mnt_path = artifact_mnt_path,
        model_version = model_version,
    ).after(train_task)
    kubernetes.mount_pvc(evaluate_task, pvc_name="data-pvc", mount_path="/data")
    kubernetes.mount_pvc(evaluate_task, pvc_name="model-pvc", mount_path="/model")
    kubernetes.mount_pvc(evaluate_task, pvc_name="artifacts-pvc", mount_path="/artifact")
    kubernetes.set_image_pull_policy(evaluate_task, "Never")

kfp.compiler.Compiler().compile(pipeline, "seoul-traffic-mlp-pipeline.yaml")