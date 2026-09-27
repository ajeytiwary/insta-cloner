from __future__ import annotations
import os,time
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw
from .config import resolve_image_model

def _dry_image(path,concept,width,height):
    img=Image.new("RGB",(width,height),(32,32,32)); d=ImageDraw.Draw(img)
    d.multiline_text((40,40),"DRY RUN\n"+str(concept.get("title",""))+"\n\n"+str(concept.get("prompt",""))[:220],fill=(235,235,235)); img.save(path)

def _download(url,path):
    import requests
    r=requests.get(url,timeout=120); r.raise_for_status(); path.write_bytes(r.content)

def _api_generate(concept,spec,path,width,height):
    import requests
    prompt=concept.get("prompt","")
    if spec["kind"]=="krea_api":
        key=os.environ.get("KREA_API_KEY")
        if not key: raise RuntimeError("Set KREA_API_KEY.")
        r=requests.post(spec["endpoint"],headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
          json={"prompt":prompt,"aspect_ratio":"4:5" if height>width else "1:1","resolution":"1K","creativity":"medium"},timeout=120)
        r.raise_for_status(); data=r.json(); job=data.get("job_id") or data.get("id")
        if not job: raise RuntimeError(f"Krea API did not return a job id: {data}")
        for _ in range(120):
            j=requests.get(f"https://api.krea.ai/jobs/{job}",headers={"Authorization":f"Bearer {key}"},timeout=30); j.raise_for_status(); payload=j.json()
            url=payload.get("result",{}).get("url") or payload.get("url") or payload.get("image_url")
            if url: _download(url,path); return
            if str(payload.get("status","")).lower() in {"failed","error"}: raise RuntimeError(str(payload))
            time.sleep(2)
        raise TimeoutError("Krea generation timed out.")
    if spec["kind"]=="ideogram_api":
        key=os.environ.get("IDEOGRAM_API_KEY")
        if not key: raise RuntimeError("Set IDEOGRAM_API_KEY.")
        r=requests.post(spec["endpoint"],headers={"Api-Key":key,"Content-Type":"application/json"},
          json={"prompt":prompt,"aspect_ratio":"4x5" if height>width else "1x1"},timeout=180); r.raise_for_status(); data=r.json()
        items=data.get("data") or []; url=items[0].get("url") if items else data.get("url")
        if not url: raise RuntimeError(f"Ideogram API did not return an image URL: {data}")
        _download(url,path); return
    raise ValueError(f"Unsupported API backend: {spec['kind']}")

def _local_diffusers(concepts,spec,output_dir,width,height,seed):
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc: raise RuntimeError('Install generation dependencies: pip install -e ".[generate]"') from exc
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected.")
    dtype=torch.float16 if spec.get("dtype")=="float16" else torch.bfloat16
    pipe=DiffusionPipeline.from_pretrained(spec["model_id"],torch_dtype=dtype)
    try: pipe.enable_model_cpu_offload()
    except Exception: pipe.to("cuda")
    for method in ("enable_vae_slicing","enable_vae_tiling"):
        if hasattr(pipe,method): getattr(pipe,method)()
    made=[]
    for i,c in enumerate(concepts):
        call={"prompt":c.get("prompt",""),"width":width,"height":height,"generator":torch.Generator(device="cpu").manual_seed(seed+i)}
        if "steps" in spec: call["num_inference_steps"]=spec["steps"]
        if "guidance_scale" in spec: call["guidance_scale"]=spec["guidance_scale"]
        im=pipe(**call).images[0]; p=output_dir/f"{i+1:03d}.png"; im.save(p); made.append(p)
    return made

def generate_images(concepts:list[dict[str,Any]],output_dir:Path,media_dir:Path,model_name:str="sdxl",
                    custom_model_id:str|None=None,custom_kind:str="diffusers",width:int=768,height:int=1024,
                    seed:int=42,dry_run:bool=False,**_:Any)->list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); spec=resolve_image_model(model_name,custom_model_id,custom_kind)
    if dry_run:
        made=[]
        for i,c in enumerate(concepts): p=output_dir/f"{i+1:03d}.png"; _dry_image(p,c,width,height); made.append(p)
        return made
    if spec["kind"]=="comfyui": raise RuntimeError("This preset requires the ComfyUI backend.")
    if spec["kind"] in {"krea_api","ideogram_api"}:
        made=[]
        for i,c in enumerate(concepts): p=output_dir/f"{i+1:03d}.png"; _api_generate(c,spec,p,width,height); made.append(p)
        return made
    return _local_diffusers(concepts,spec,output_dir,width,height,seed)
