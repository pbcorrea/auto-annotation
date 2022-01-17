# Automated labelling

Running automated annotations on CVAT, deployed on a GCP VM.

# 1. Launch GCP Virtual Machine

## 1.1. Create VM
```bash
gcloud compute instances create cvat \
    --project flawless-agency-337617 \
    --machine-type n1-standard-2 \
    --zone us-east1-c \
    --boot-disk-size 200 \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --image-family common-cu110 \
    --image-project deeplearning-platform-release \
    --maintenance-policy TERMINATE --restart-on-failure
```


# 1.2. Enable OS Login and setup permissions

```bash
gcloud config set project flawless-agency-337617
gcloud compute instances add-iam-policy-binding cvat --zone=us-east1-c --member='user:pbcorrea@rmc.cl' --role='roles/compute.securityAdmin'
```


# 1.3. Access through SSH
```bash
gcloud compute ssh --project=flawless-agency-337617 --zone=us-east1-c cvat
```

## 1.4. (Optional) Enable public access

```bash
gcloud compute --project=flawless-agency-337617 firewall-rules create jupyterlab --description="jupyterlab public access" --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=all --source-ranges=0.0.0.0/0
```

# 2. Install CVAT

[Instructions](https://openvinotoolkit.github.io/cvat/docs/administration/basics/installation/#ubuntu-1804-x86_64amd64)


## 2.1. Install pre-requisites
```bash
sudo apt-get update
sudo apt-get --no-install-recommends install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg-agent \
  software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"
sudo apt-get update
sudo apt-get --no-install-recommends install -y docker-ce docker-ce-cli containerd.io
```

## 2.2. Enable docker access

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

## 2.3. Install docker-compose

```bash
sudo apt-get --no-install-recommends install -y python3-pip python3-setuptools
python3 -m pip install setuptools docker-compose
```

## 2.4. Clone and install CVAT

```bash
sudo apt-get --no-install-recommends install -y git
git clone https://github.com/opencv/cvat
cd cvat
```

## 2.5. Stop jupyterlab 

To avoid port 8080 use conflict

```bash
sudo systemctl stop jupyter.service
```

## 2.6. Launch docker-compose

```bash
export CVAT_HOST=<GCP_VM_PUBLIC_IP>
docker-compose up -d
docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
```


# 3. Run serverless annotation using detectron2 and nuclio

## 3.1. Create serverless function.

See `serverless_inference.py` for a reference function.

## 3.2. Launch docker-compose with nuclio service.

