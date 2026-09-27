# insta-cloner

Analyze the visual language of an Instagram profile and turn it into a reusable **inspiration profile** for original content.

The tool intentionally separates:

- **visual language**: palette, lighting, composition, recurring framing, scene types;
- **pose/composition references**: extracted structural references;
- **identity**: not copied by this project;
- **generation**: optional downstream stage.

It is designed for inspiration and visual-system analysis, not impersonation.

## What works now

1. Accept an Instagram profile URL or username.
2. Download a bounded sample of recent posts with Instaloader.
3. Extract video keyframes from downloaded Reels/videos.
4. Analyze images with deterministic image statistics.
5. Optionally use CLIP Interrogator for richer text descriptions.
6. Aggregate the sample into `style_profile.json`.
7. Produce `generation_brief.md` for your local LLM or image workflow.
8. Optionally extract pose maps with a ControlNet-compatible annotator when the pose extra is installed.

## Install

Use Python 3.10-3.13.

```bash
git clone https://github.com/ajeytiwary/insta-cloner.git
cd insta-cloner

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows

pip install -e .
```

For CLIP Interrogator:

```bash
pip install -e ".[clip]"
```

For optional pose extraction:

```bash
pip install -e ".[pose]"
```

## Usage

Public profile:

```bash
insta-cloner analyze https://www.instagram.com/PROFILE/ --max-posts 12
```

If Instagram requires authentication, first create/reuse an Instaloader session:

```bash
instaloader --login YOUR_INSTAGRAM_USERNAME
insta-cloner analyze PROFILE --max-posts 12 --login YOUR_INSTAGRAM_USERNAME
```

Enable CLIP interrogation:

```bash
insta-cloner analyze PROFILE --max-posts 12 --clip
```

Enable pose maps:

```bash
insta-cloner analyze PROFILE --max-posts 12 --pose
```

Output:

```text
output/PROFILE/
├── media/
├── frames/
├── poses/
├── sample_manifest.json
├── per_image_analysis.json
├── style_profile.json
└── generation_brief.md
```

## Local-model workflow

Feed `style_profile.json` and `generation_brief.md` to your local model and ask it to create original concepts that preserve high-level characteristics while changing subjects, scenes, garments, props, and exact compositions.

A downstream image pipeline can use:

- text prompt from the brief;
- IP-Adapter for broad reference-image conditioning;
- ControlNet/OpenPose/DWPose-compatible pose maps for structure;
- your own character or identity model, if applicable.

The project does **not** train on or reproduce another creator's identity.

## Notes

Instagram can rate-limit or require login. The project uses Instaloader rather than undocumented browser scraping. Keep sample sizes modest.

CLIP Interrogator downloads model weights on first use. Pose annotators may also download weights. The base analyzer works without either optional dependency.
