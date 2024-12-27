# DiffusionHeadshot

This repository is a quick demo fine-tuning a diffusion model FLUX.1 [dev] with
QLoRA. The goal of this work was threefold:

1. I need a professional photo for job applications.
2. I need a portfolio for job applications.
3. I want to make Christmas cards, etc. with my family in them.

So this is an experiment in LoRA training with restricted resources.

## Experiments

Experiments is included near the top so that results can be viewed before
diving into all of the ideas behind everything.

* [Experiment 1](experiments/experiment1/README.md)

## Starting Point (Limitations)

This work makes small modifications to [the LoRA flux quantization research example](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/flux_lora_quantization).

Flux1.dev was selected because it's a current state-of-the-art (open source)
diffusion model for image generation. Unfortunately, I do not have a 4090 which
most of the examples are trained on. There are a few other sources that provide
quantized models already but it's not clear whether they fit on the GPUs that
I own (3060 RTX (12GiB GDDR6) or RX 5500 XT OC Gaming (8GiB GDDR5)).

So here we use BitsAndBytes to quantize the model down to a manageable size for
my card and construct our own dataset to fine-tune on.

I did not want to use a UI such as [Automatic1111 web
ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) or
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) because they hide the
process of training and understanding the actual code that goes into making the
model.

## Low-Rank Adaptation (LoRA)

[Low Rank Adaptation](https://arxiv.org/abs/2106.09685) is a parameter
efficient method for fine-tuning LLMs (PEFT) and now diffusion models as well.
It works by decomposing the update matrix $\delta W$ into two lower-rank
matrices so that instead of tuning $`M\cdot N`$ parameters, you only have to
tune $`M\cdot r+r\cdot N`$ parameters where $`r\ll M`$ and $`r\ll N`$. It
appears that the default value for $`r=2`$ in general. According to the paper,
problems are typically have a low "intrinsic rank" meaning that most of the
parameters are not used in each individual problem.

## Quantization

Since my 3060 RTX is a part of the nVidia Ampere architecture, the BitsAndBytes
library supports 4-bit quantization with NF4 (Normal Float 4) with empirical
results that it performs better than fp4 quantization in [the QLoRA
paper](https://arxiv.org/abs/2305.14314v1).

## Family Dataset

I manually tagged images of myself (JRussell), my wife (ARussell), and our son
(CRussell) and implemented a wrapper to make an image dataset, updating the
example code to load my dataset instead of the original dataset. Due to privacy
concerns, I won't upload the images from the dataset but will provide the
textual descriptions as they are relevant to understanding the tuned model.

### Problems

Most LoRA how-to examples imply that you should describe everything in the
scene except for the one thing that you want to train the LoRA adapter for.
Most examples also train their adapter for solely a single concept. I wanted to
be able to generate images with my entire family in them and therefore desired
to have multiple triggers.

I also elected to manually caption instead of generate captions with tools like
wd14 or BLIP because then I can understand the impact of each caption better
with respect to model generation.
