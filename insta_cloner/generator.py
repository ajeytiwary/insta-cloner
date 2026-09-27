from __future__ import annotations
import os,time
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw
from .config import resolve_image_model

def _dry_image(path,concept,width,height):
    img=Image.new("RGB",(width,height),(32,32,32)); ImageDraw.Draw(img).multiline_text((40,40),"DRY RUN\n"+str(concept.get("title",""))+"\n\n"+str(concept.get("prompt",""))[:220],fill=(235,235,235)); img.save(path)

def _download(url,path):
    import requests
    r=requests.get(url,timeout=120); r.raise_for_status(); path.write_bytes(r.content)

def _api_generate(concept,spec,path,width,height):
    import requests
    prompt=concept.get("prompt","")
    if spec["kind"]=="krea_api":
        key=os.environ.get("KREA_API_KEY")
        if not key: raise RuntimeError("Set KREA_API_KEY.")
        r=requests.post(spec["endpoint"],headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"prompt":prompt,"aspect_ratio":"4:5" if height>width else "1:1","resolution":"1K","creativity":"medium"},timeout=120); r.raise_for_status(); data=r.json(); job=data.get("job_id") or data.get("id")
        for _ in range(120):
            j=requests.get(f"https://api.krea.ai/jobs/{job}",headers={"Authorization":f"Bearer {key}"},timeout=30); j.raise_for_status(); payload=j.json(); url=payload.get("result",{}).get("url") or payload.get("url") or payload.get("image_url")
            if url: _download(url,path); return
            time.sleep(2)
        raise TimeoutError("Krea generation timed out.")
    if spec["kind"]=="ideogram_api":
        key=os.environ.get("IDEOGRAM_API_KEY")
        if not key: raise RuntimeError("Set IDEOGRAM_API_KEY.")
        r=requests.post(spec["endpoint"],headers={"Api-Key":key,"Content-Type":"application/json"},json={"prompt":prompt,"aspect_ratio":"4x5" if height>width else "1x1"},timeout=180); r.raise_for_status(); data=r.json(); items=data.get("data") or []; url=items[0].get("url") if items else data.get("url")
        if not url: raise RuntimeError(f"Ideogram API did not return an image URL: {data}")
        _download(url,path); return
    raise ValueError(spec["kind"])

def _reference(media_dir:Path,index:int=0):
    paths=[p for p in sorted(media_dir.rglob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    return Image.open(paths[index%len(paths)]).convert("RGB") if paths else None

def _conditioned(concepts,spec,output_dir,media_dir,width,height,seed,ip_adapter,ip_scale,ip_repo,ip_weight,controlnet,control_scale,control_model,pose_dir):
    import torch
    from diffusers import AutoPipelineForText2Image,ControlNetModel,StableDiffusionXLControlNetPipeline
    model_id=spec["model_id"]
    if controlnet:
        cn=ControlNetModel.from_pretrained(control_model,torch_dtype=torch.float16)
        pipe=StableDiffusionXLControlNetPipeline.from_pretrained(model_id,controlnet=cn,torch_dtype=torch.float16,use_safetensors=True)
    else:
        pipe=AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float16,use_safetensors=True)
    if ip_adapter:
        pipe.load_ip_adapter(ip_repo,subfolder="sdxl_models",weight_name=ip_weight); pipe.set_ip_adapter_scale(ip_scale)
    pipe.enable_model_cpu_offload()
    poses=[p for p in sorted((pose_dir or Path()).glob("*.png"))] if pose_dir else []
    made=[]
    for i,c in enumerate(concepts):
        call={"prompt":c.get("prompt",""),"width":width,"height":height,"num_inference_steps":spec.get("steps",25),"generator":torch.Generator(device="cpu").manual_seed(seed+i)}
        ref=_reference(media_dir,i)
        if ip_adapter and ref is not None: call["ip_adapter_image"]=ref
        if controlnet:
            if not poses: raise RuntimeError("ControlNet enabled but no pose maps were extracted. Enable pose extraction.")
            call["image"]=Image.open(poses[i%len(poses)]).convert("RGB"); call["controlnet_conditioning_scale"]=control_scale
        im=pipe(**call).images[0]; p=output_dir/f"{i+1:03d}.png"; im.save(p); made.append(p)
    return made

def _local(concepts,spec,output_dir,width,height,seed,steps=None,guidance_scale=None,low_vram=True):

    import torch
    from diffusers import DiffusionPipeline
    from .runtime import apply_memory_optimizations
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected.")
    pipeline_name=spec.get("pipeline")
    if pipeline_name=="qwen_image_2_1":
        from diffusers import QwenImage21Pipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        load_kwargs={"dtype":torch.bfloat16}
        if low_vram:
            load_kwargs["quantization_config"]=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit":True,"bnb_4bit_quant_type":"nf4","bnb_4bit_compute_dtype":torch.bfloat16},
                components_to_quantize=["transformer","text_encoder"],
            )
        pipe=QwenImage21Pipeline.from_pretrained(spec["model_id"],**load_kwargs)
    elif pipeline_name=="krea2":
        from diffusers import Krea2Pipeline
        pipe=Krea2Pipeline.from_pretrained(spec["model_id"],torch_dtype=torch.bfloat16)
    else:
        pipe=DiffusionPipeline.from_pretrained(spec["model_id"],torch_dtype=torch.bfloat16)
    pipe=apply_memory_optimizations(pipe,low_vram=low_vram)
    made=[]
    for i,c in enumerate(concepts):
        call={"prompt":c.get("prompt",""),"width":width,"height":height,"generator":torch.Generator(device="cpu").manual_seed(seed+i)}
        chosen_steps=steps if steps is not None else spec.get("steps")
        chosen_guidance=guidance_scale if guidance_scale is not None else spec.get("guidance_scale")
        if chosen_steps is not None: call["num_inference_steps"]=int(chosen_steps)
        if chosen_guidance is not None and pipeline_name!="qwen_image_2_1": call["guidance_scale"]=float(chosen_guidance)
        im=pipe(**call).images[0]; p=output_dir/f"{i+1:03d}.png"; im.save(p); made.append(p)
    return made

def generate_images(concepts:list[dict[str,Any]],output_dir:Path,media_dir:Path,model_name:str="sdxl",width:int=768,height:int=1024,seed:int=42,dry_run:bool=False,
                    ip_adapter:bool=False,ip_scale:float=.45,ip_repo:str="h94/IP-Adapter",ip_weight:str="ip-adapter_sdxl.bin",
                    controlnet:bool=False,control_scale:float=.8,control_model:str="thibaud/controlnet-openpose-sdxl-1.0",pose_dir:Path|None=None,steps:int|None=None,guidance_scale:float|None=None,backend:str="auto",low_vram:bool=True,**_:Any)->list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); spec=resolve_image_model(model_name)
    if dry_run:
        made=[]
        for i,c in enumerate(concepts): p=output_dir/f"{i+1:03d}.png"; _dry_image(p,c,width,height); made.append(p)
        return made
    if backend=="comfyui" or spec["kind"]=="comfyui":
        if ip_adapter or controlnet:
            raise RuntimeError("SDXL IP-Adapter/ControlNet controls cannot be attached to this ComfyUI workflow.")
        from .comfyui import generate_comfy
        workflow_path=Path(spec.get("workflow_path",""))
        if not workflow_path.exists():
            raise RuntimeError(
                f"ComfyUI API workflow not found: {workflow_path}. "
                "Export the upstream workflow with ComfyUI 'Save (API Format)' and set workflow_path/mapping in Model configuration."
            )
        return generate_comfy(
            [c.get("prompt","") for c in concepts],
            output_dir,
            workflow_path,
            spec.get("mapping",{}),
            spec.get("comfy_url","http://127.0.0.1:8188"),
            width,height,seed,int(spec.get("steps",25)),
        )
    if spec["kind"] in {"krea_api","ideogram_api"}:
        if ip_adapter or controlnet: raise RuntimeError("IP-Adapter/ControlNet controls currently require a compatible local Diffusers SDXL backend.")
        made=[]
        for i,c in enumerate(concepts): p=output_dir/f"{i+1:03d}.png"; _api_generate(c,spec,p,width,height); made.append(p)
        return made
    if ip_adapter or controlnet:
        return _conditioned(concepts,spec,output_dir,media_dir,width,height,seed,ip_adapter,ip_scale,ip_repo,ip_weight,controlnet,control_scale,control_model,pose_dir)
    return _local(concepts,spec,output_dir,width,height,seed,steps,guidance_scale,low_vram)
