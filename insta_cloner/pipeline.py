from __future__ import annotations
from pathlib import Path
from typing import Any
from .analyzer import analyze_images,save_style_profile
from .brief import create_generation_brief
from .downloader import download_sample
from .frames import extract_keyframes
from .generator import generate_images
from .llm import generate_concepts
from .pose import extract_poses
from .utils import profile_name,write_json
from .vlm import analyze_with_vlm,merge_vlm_style
IMAGE_SUFFIXES={".jpg",".jpeg",".png",".webp"}

def run_pipeline(profile:str,output:Path=Path("output"),max_posts:int=12,login:str|None=None,use_clip:bool=False,use_pose:bool=False,video_frames:int=4,concept_count:int=12,
 llm_base_url:str="http://127.0.0.1:8080/v1",llm_model:str="local-model",llm_api_key:str="local",vlm:bool=False,vlm_url:str="http://127.0.0.1:8000/v1",vlm_model:str="Qwen/Qwen3-VL-4B-Instruct",
 generate:bool=False,image_model:str="sdxl",ip_adapter:bool=False,ip_scale:float=.45,ip_repo:str="h94/IP-Adapter",ip_weight:str="ip-adapter_sdxl.bin",
 controlnet:bool=False,control_scale:float=.8,control_model:str="thibaud/controlnet-openpose-sdxl-1.0",dry_run_llm:bool=False,dry_run_generation:bool=False)->dict[str,Any]:
    username=profile_name(profile); out=output/username; out.mkdir(parents=True,exist_ok=True)
    manifest=download_sample(username,out,max_posts=max_posts,login=login); frames=extract_keyframes(out/"media",out/"frames",per_video=video_frames)
    records=analyze_images([out/"media",out/"frames"],out,use_clip=use_clip); style=save_style_profile(records,out); vlm_results=[]
    if vlm:
        paths=sorted({p for root in [out/"media",out/"frames"] for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES})
        vlm_results=analyze_with_vlm(paths,out/"vlm_aesthetic_analysis.json",base_url=vlm_url,model=vlm_model); style=merge_vlm_style(style,vlm_results); write_json(out/"style_profile.json",style)
    create_generation_brief(username,style,out/"generation_brief.md")
    need_pose=use_pose or controlnet
    poses=extract_poses([out/"media",out/"frames"],out/"poses") if need_pose else []
    concepts=generate_concepts(style,out/"concepts.json",count=concept_count,base_url=llm_base_url,model=llm_model,api_key=llm_api_key,dry_run=dry_run_llm)
    images=generate_images(concepts,out/"generated",out/"media",model_name=image_model,dry_run=dry_run_generation,
      ip_adapter=ip_adapter,ip_scale=ip_scale,ip_repo=ip_repo,ip_weight=ip_weight,controlnet=controlnet,control_scale=control_scale,control_model=control_model,pose_dir=out/"poses") if generate else []
    return {"username":username,"output":str(out),"posts":len(manifest),"frames":len(frames),"analyzed":len(records),"vlm_analyses":len(vlm_results),"poses":len(poses),
      "concepts":len(concepts),"generated":len(images),"image_model":image_model,"ip_adapter":ip_adapter,"controlnet":controlnet}
