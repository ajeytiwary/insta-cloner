# insta-cloner

Local pipeline for turning an Instagram account's **visual language** into original content concepts and optional generated images.

## Pipeline

Instagram URL -> Instaloader sample -> image/Reel keyframes -> visual statistics + optional CLIP -> style_profile.json -> local OpenAI-compatible LLM -> concepts.json -> optional Diffusers SDXL + IP-Adapter -> generated/

Identity copying is deliberately kept separate: the pipeline extracts broad aesthetics, composition tendencies and optional structural pose references, then asks the local model to change the exact person, location, wardrobe, props and composition.

## Install

Python 3.10-3.13:

    git clone https://github.com/ajeytiwary/insta-cloner.git
    cd insta-cloner
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[llm]"

For everything:

    pip install -e ".[full]"

## llama.cpp

Start any OpenAI-compatible llama.cpp server, for example on port 8080. The app defaults to:

    http://127.0.0.1:8080/v1

The model name is passed through to the server, so set --llm-model to whatever your server exposes.

## One-command workflow

Concepts only:

    insta-cloner clone https://www.instagram.com/PROFILE/ \
      --max-posts 12 \
      --clip \
      --concepts 12 \
      --llm-url http://127.0.0.1:8080/v1 \
      --llm-model local-model

Concepts + images:

    insta-cloner clone PROFILE \
      --max-posts 12 \
      --clip \
      --concepts 12 \
      --generate \
      --image-model stabilityai/stable-diffusion-xl-base-1.0

Add IP-Adapter broad visual conditioning:

    insta-cloner clone PROFILE \
      --concepts 12 \
      --generate \
      --ip-adapter-repo h94/IP-Adapter

For Instagram login-required profiles, first create an Instaloader session:

    instaloader --login YOUR_INSTAGRAM_USERNAME

Then add:

    --login YOUR_INSTAGRAM_USERNAME

## RTX 3060 12 GB

The default SDXL generator uses fp16, model CPU offload, VAE slicing and VAE tiling. Default output is 768x1024 and 25 steps. This favors fitting into 12 GB rather than maximum speed.

CLIP Interrogator and pose extraction are optional because they add models/VRAM pressure. If memory is tight, run analysis first and image generation as a separate invocation/process.

## Validation without models

The LLM and image stages both have dry-run modes. They exercise orchestration without downloading model weights:

    insta-cloner clone PROFILE \
      --max-posts 2 \
      --dry-run-llm \
      --generate \
      --dry-run-generation

## Output

    output/PROFILE/
    ├── media/
    ├── frames/
    ├── poses/
    ├── sample_manifest.json
    ├── per_image_analysis.json
    ├── style_profile.json
    ├── generation_brief.md
    ├── concepts.json
    └── generated/
        ├── 001.png
        └── ...

## Components

- Instaloader: Instagram sampling/download.
- CLIP Interrogator: optional semantic visual descriptions.
- OpenPose annotator: optional pose maps.
- Any OpenAI-compatible local LLM: concept/art-direction generation.
- Diffusers SDXL: default local image backend.
- IP-Adapter: optional reference-image conditioning.

The image backend is isolated in generator.py so Qwen-Image, FLUX or another Diffusers-compatible backend can be added without changing the downloader/analyzer/LLM stages.


## One-command GPU cloud deployment

The WebUI can run without ComfyUI in the included NVIDIA container. Model downloads are cached under `/data/huggingface`.

### AWS

Default target: EC2 G6. Override `INSTANCE_TYPE` to choose another GPU instance family.

```bash
AWS_REGION=eu-central-1 INSTANCE_TYPE=g6.2xlarge ALLOWED_CIDR="$(curl -s https://checkip.amazonaws.com)/32" ./deploy.sh aws
```

The CloudFormation stack creates the GPU VM, persistent root storage, SSM role and WebUI security group. For anything beyond temporary testing, restrict `ALLOWED_CIDR` rather than exposing port 7860 globally.

### GCP

Default target: `g2-standard-8` with one NVIDIA L4-class GPU. GPU quota and the selected machine type must be available in the chosen zone.

```bash
PROJECT_ID=my-project ZONE=europe-west4-a MACHINE_TYPE=g2-standard-8 ./deploy.sh gcp
```

Both deployments accept larger GPU machine types without application changes. Large checkpoints should use a sufficiently large disk; the default is 250 GB.
