# End-to-end validation

Paste terminal output / screenshots for each step into the final PR description.

## 1. Local Docker

Quote the `-v` mounts - `$(pwd)` here is under a path with spaces. Host port 8080 is used
by a Windows service on this machine, so the examples map to host port 8000; the container
still serves on 8080.

```bash
mkdir -p data checkpoints

docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run -d --name mlops-serve -p 8000:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1

# a sample image to POST
python -c "from torchvision import datasets; datasets.CIFAR10('data', download=True)[1][0].save('test_image.png')"

sleep 8
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "image=@test_image.png"
docker logs mlops-serve
docker rm -f mlops-serve
```

## 2. Kubernetes

Local cluster with kind (single node, reuses the WSL Docker engine):

```bash
kind create cluster --name mlops
kind load docker-image mlops-train:v1 --name mlops
kind load docker-image mlops-serve:v1 --name mlops

# metrics-server so the HPA can read CPU (kind needs --kubelet-insecure-tls)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Run training as a Job:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl get pods -n ml-training -w
kubectl logs -f job/train-classifier -n ml-training
# let it finish, or stop once a checkpoint is written to the PVC:
#   kubectl delete job train-classifier -n ml-training
```

Deploy serving:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment/model-serving -n ml-training

kubectl get pods,deploy,svc,hpa,pvc -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8000:80 -n ml-training
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "image=@test_image.png"
```

On a cloud cluster with GPU nodes, apply `k8s/training-job-gpu.yaml` instead of
`k8s/training-job.yaml` (see README).

## Reflection

_What was the most challenging part?_

The model code was the easy part. The friction was all in the boundaries between
environments - local shell, Docker, and Kubernetes - where the same script has to
behave the same way with different filesystems, ports, and process lifecycles.

The single most useful design decision was making the two images separate rather
than one image with a mode flag. Training needs torchvision's dataset and
augmentation machinery; serving only needs inference plus the web framework. Split
requirements files, a slim base, and a non-root user keep the serving image small
and give it a smaller attack surface, which is the point of a serving container.

Getting the model artifact from the training workload to the serving workload was
the part that most changed how I think about this. Locally it is just a bind mount.
On Kubernetes the Job writes `classifier_v1.pt` to a PersistentVolumeClaim and the
Deployment mounts the same PVC read-only. That means no model registry or object
store is needed for a single-cluster setup, but it also means the serving pods are
useless until the Job has produced at least one checkpoint - so `serve.py` had to
treat "no checkpoint yet" as a normal 503 state instead of crashing. That one
choice is what makes the readiness probe and the CI smoke test behave sensibly
before any training has happened.

Smaller things that cost real time: `docker run -v $(pwd)/...` silently splitting
on the space in the repo path; host port 8080 being held by a Windows service, so
every local and port-forward example had to move to 8000 while the container kept
`EXPOSE 8080` as the spec requires; the training loop only logging per-epoch, which
made early runs look hung until I checked `docker stats`; and the Job's
`requests: cpu 2 / memory 4Gi` not fitting a default local cluster node until it
was given more resources. None of these are hard once identified, but each is the
kind of thing that only shows up when you actually run the full path end to end,
which is the main lesson: the pipeline is only "done" when every stage has been
executed against the real runtime, not just the code reviewed.

<!-- fill in from the actual runs before submitting -->
- Docker training: reached epoch N, val_accuracy ~0.XX, checkpoint saved.
- Docker serving: /health 200, /predict returned "<class>".
- Kubernetes: Job completed / stopped after checkpoint; serving Deployment 2/2 ready;
  /predict via port-forward returned "<class>".
