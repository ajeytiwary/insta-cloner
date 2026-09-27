#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
ZONE="${ZONE:-europe-west4-a}"
INSTANCE_NAME="${INSTANCE_NAME:-insta-cloner-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
DISK_SIZE="${DISK_SIZE:-250GB}"
REPO_URL="${REPO_URL:-https://github.com/ajeytiwary/insta-cloner.git}"
REPO_REF="${REPO_REF:-main}"
WEB_PORT="${WEB_PORT:-7860}"
WEB_USER="${WEB_USER:-admin}"
WEB_PASSWORD="${WEB_PASSWORD:?Set WEB_PASSWORD (12+ characters)}"

gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com

cat > /tmp/insta-cloner-startup.sh <<EOF
#!/usr/bin/env bash
set -euxo pipefail
apt-get update
apt-get install -y docker.io git curl
systemctl enable --now docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
mkdir -p /opt/insta-cloner /data
git clone --branch "$REPO_REF" "$REPO_URL" /opt/insta-cloner
cd /opt/insta-cloner
docker build -t insta-cloner .
docker run --rm --gpus all -v /data:/data insta-cloner insta-cloner gpu-smoke --output /data/smoke
docker run -d --restart unless-stopped --gpus all --name insta-cloner \
  -e INSTA_CLONER_USER="$WEB_USER" \
  -e INSTA_CLONER_PASSWORD="$WEB_PASSWORD" \
  -p "$WEB_PORT":7860 \
  -v /data:/data \
  insta-cloner
EOF

gcloud compute firewall-rules describe insta-cloner-web >/dev/null 2>&1 || \
gcloud compute firewall-rules create insta-cloner-web \
  --allow="tcp:$WEB_PORT" \
  --target-tags=insta-cloner

gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --boot-disk-size="$DISK_SIZE" \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --tags=insta-cloner \
  --metadata-from-file=startup-script=/tmp/insta-cloner-startup.sh

IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "WebUI: http://$IP:$WEB_PORT"
