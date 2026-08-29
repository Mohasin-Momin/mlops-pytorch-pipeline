# End-to-end validation

Paste terminal output / screenshots for each step into the final PR description.

## 1. Local Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## 2. Kubernetes

```bash
minikube image load mlops-train:v1
minikube image load mlops-serve:v1

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/train-classifier -n ml-training
kubectl wait --for=condition=complete job/train-classifier -n ml-training --timeout=1800s

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Reflection (300-500 words)

_What was the most challenging part?_

Draft notes to expand before submission:

- Keeping the training and serving images small and separate. The training image
  needs torchvision datasets and augmentation; the serving image only needs
  inference plus the web framework. Splitting the requirements files and using a
  slim base with a non-root user kept the serving image lean.
- Path and import handling across three run contexts (local, Docker, Kubernetes).
  The config path is resolved in a fixed order and `PYTHONPATH=/app/src` is set
  in the images so the flat `import model` / `import dataset` works everywhere.
- Storage wiring on Kubernetes. The Job writes the checkpoint to a PVC and the
  serving Deployment mounts the same PVC read-only, so the two workloads share
  the artifact without a registry or object store.
- Probes and rolling updates. Readiness needs an initial delay because model load
  takes a few seconds; `maxUnavailable: 0` keeps at least two pods serving during
  an update.
