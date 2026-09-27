#!/usr/bin/env bash
set -euo pipefail

CLOUD="${1:-}"

case "$CLOUD" in
  aws)
    STACK="${STACK:-insta-cloner-gpu}"
    REGION="${AWS_REGION:-eu-central-1}"
    INSTANCE_TYPE="${INSTANCE_TYPE:-g6.2xlarge}"
    ALLOWED_CIDR="${ALLOWED_CIDR:-0.0.0.0/0}"
    WEB_USER="${WEB_USER:-admin}"
    WEB_PASSWORD="${WEB_PASSWORD:?Set WEB_PASSWORD (12+ characters)}"

    aws cloudformation deploy \
      --region "$REGION" \
      --stack-name "$STACK" \
      --template-file deploy/aws/cloudformation.yaml \
      --capabilities CAPABILITY_NAMED_IAM \
      --parameter-overrides \
        InstanceType="$INSTANCE_TYPE" \
        AllowedCidr="$ALLOWED_CIDR" \
        WebUser="$WEB_USER" \
        WebPassword="$WEB_PASSWORD"

    aws cloudformation describe-stacks \
      --region "$REGION" \
      --stack-name "$STACK" \
      --query 'Stacks[0].Outputs' \
      --output table
    ;;
  gcp)
    exec bash deploy/gcp/deploy.sh
    ;;
  *)
    echo "Usage: ./deploy.sh aws|gcp"
    echo 'AWS: WEB_PASSWORD="choose-a-long-password" AWS_REGION=eu-central-1 INSTANCE_TYPE=g6.2xlarge ALLOWED_CIDR="YOUR_IP/32" ./deploy.sh aws'
    echo 'GCP: WEB_PASSWORD="choose-a-long-password" PROJECT_ID=my-project MACHINE_TYPE=g2-standard-8 ZONE=europe-west4-a ./deploy.sh gcp'
    exit 2
    ;;
esac
