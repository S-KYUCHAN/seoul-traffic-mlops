# seoul-traffic-mlops

Kubeflow Pipelines(KFP) + Katib(HPO)로 모델을 학습/평가하고, **Argo CD 기반 GitOps로 KServe InferenceService의 모델 버전을 자동 전개**하는 예제 프로젝트입니다.

- **Train/Eval**: KFP 파이프라인 실행 → 모델 아티팩트가 PVC(`/model/<MODEL_VERSION>/model.joblib`)에 저장
- **Deploy**: GitOps repo에서 `MODEL_VERSION`만 변경 → Argo CD가 자동 Sync → KServe가 `storageUri`를 새 버전으로 교체하여 롤아웃

---

## Architecture

1. (Data) 월별 CSV를 `dataset-pvc`에 업로드
2. (Pipeline)
   - `featurization` → 학습/테스트 CSV 생성
   - `tuning-with-katib` → Katib 실험 실행 후 best params로 최종 학습 → `/model/<MODEL_VERSION>/model.joblib`
   - `evaluate-model` → `/artifact/<MODEL_VERSION>/score.json`
3. (GitOps Deploy)
   - `gitops/overlays/dev/patch-model-version.yaml`에서 모델 버전만 변경
   - Argo CD Application이 `gitops/overlays/dev`를 감시하고 자동 Sync
   - KServe InferenceService의 `spec.predictor.model.storageUri`가 새 버전으로 업데이트되어 서빙 롤아웃

---

## Repo Layout

```text
.
├─ pipelines/                # KFP pipeline YAML / pipeline build code
├─ gitops/
│  ├─ base/
│  │  ├─ inferenceservice.yaml
│  │  └─ kustomization.yaml
│  └─ overlays/
│     └─ dev/
│        ├─ kustomization.yaml
│        └─ patch-model-version.yaml
└─ scripts/                  # (옵션) 모델 버전 bump / PR 자동화 스크립트
```

---

## Prerequisites

- Kubernetes + Kubeflow (KFP, Katib) 구성
- KServe + Knative/Istio 구성
- PVC 3개 (예시)
  - `dataset-pvc` : 원천 CSV 및 전처리 데이터
  - `model-pvc` : 학습된 모델 저장
  - `artifact-pvc` : 평가 결과(메트릭 등) 저장
- Argo CD 설치 (GitOps 자동 Sync)

---

## 1) KServe 리소스(로컬) 확인

`gitops/base/inferenceservice.yaml`은 InferenceService 리소스를 정의합니다.
현재는 `storageUri`를 PVC 기반으로 쓰고, 버전은 overlay에서 patch로 바꿉니다.

- Base의 `storageUri`(placeholder):
  - `pvc://model-pvc/MODEL_VERSION/`

Overlay는 base를 참조하고 patch를 적용합니다.

---

## 2) GitOps(Kustomize)로 InferenceService 적용

로컬에서 kustomize apply로 먼저 동작 확인:

```bash
kubectl apply -k gitops/overlays/dev/
kubectl get isvc -n kubeflow-user-example-com seoul-traffic-model
```

gitops/overlays/dev/patch-model-version.yaml의 값이 실제 적용됩니다. 

---

## 3) Argo CD Application 생성

Argo CD가 이 repo의 gitops/overlays/dev 경로를 감시하도록 Application을 생성합니다.

예시(argocd-app-dev.yaml):

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: seoul-traffic-dev
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/S-KYUCHAN/seoul-traffic-mlops
    targetRevision: main
    path: gitops/overlays/dev
  destination:
    server: https://kubernetes.default.svc
    namespace: kubeflow-user-example-com
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

적용:
```bash
kubectl apply -f argocd-app-dev.yaml
```

이제부터는 repo에 push만 하면 Argo CD가 자동으로 Sync합니다(OutOfSync 발생 시 자동 복구).

---

## 4) 모델 버전 전개(배포) 방식

gitops/overlays/dev/patch-model-version.yaml에서 MODEL_VERSION 부분을 새 버전으로 변경 후 push:
```yaml
spec:
  predictor:
    model:
      storageUri: pvc://model-pvc/2025-05-01/
```
> patch 파일은 base의 storageUri를 덮어쓰는 용도입니다.

push 하면 Argo CD가 감지 → InferenceService 업데이트 → 새 revision 롤아웃

---

## 5) 검증(Verification)

### 5-1) InferenceService가 새 버전을 가리키는지 확인
```bash
NS=kubeflow-user-example-com
kubectl get isvc -n $NS seoul-traffic-model -o jsonpath='{.spec.predictor.model.storageUri}'; echo
```
> KServe 버전에 따라 storageUri 필드는 spec.predictor.model.storageUri 또는 spec.predictor.sklearn.storageUri로 표현될 수 있으며, 본 프로젝트는 GitOps patch로 해당 값을 업데이트합니다.

### 5-2) 예측 호출(클러스터 내부)

환경에 따라 외부 URL이 막혀있을 수 있어, 일반적으로는
 - kubectl port-forward로 predictor private service를 포워딩하거나
 - Notebook Pod에서 내부 호출합니다.

(예: port-forward)
```bash
NS=kubeflow-user-example-com
SVC=$(kubectl -n $NS get ksvc seoul-traffic-model-predictor -o jsonpath='{.status.latestReadyRevisionName}')
kubectl -n $NS port-forward svc/${SVC}-private 9999:8012
curl -X POST http://localhost:9999/v1/models/seoul-traffic-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 5.0, 10.0]]}'
```

---

## Note

- 이 repo는 “모델 이미지 빌드/푸시” 대신 PVC 기반 모델 로딩(storageUri) 패턴을 사용합니다. 
