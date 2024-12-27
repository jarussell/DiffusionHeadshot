#!/usr/bin/env python3

from diffusers import AutoPipelineForText2Image, FluxTransformer2DModel, BitsAndBytesConfig
import torch

ckpt_id = "black-forest-labs/FLUX.1-dev"
bnb_4bit_compute_dtype = torch.float16
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)
transformer = FluxTransformer2DModel.from_pretrained(
    "fused_transformer",
    quantization_config=nf4_config,
    torch_dtype=bnb_4bit_compute_dtype,
)
pipeline = AutoPipelineForText2Image.from_pretrained(
    ckpt_id, transformer=transformer, torch_dtype=bnb_4bit_compute_dtype
)
pipeline.enable_model_cpu_offload()

image = pipeline(
    "a professional photograph of JRussell with a clean-shaven head, wearing a suit and tie",
	num_inference_steps=28,
	guidance_scale=3.5,
	 height=768
).images[0]
image.save("JRussell_headshot.png")

image = pipeline(
    "A christmas card with adult ARussell, child CRussell, and adult JRussell lit by the warm glow of the fireplace",
	num_inference_steps=28,
	guidance_scale=3.5,
	 height=768
).images[0]
image.save("family_christmas.png")
