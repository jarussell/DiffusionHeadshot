# LLMHeadshot

This repository is a quick demo fine-tuning a multimodal Large Language Model
(LLM).

The model was selected because of limitations on local compute and likely
impacts the output results.

# Low-Rank Adaptation (LoRA)

[Low Rank Adaptation](https://arxiv.org/abs/2106.09685) is a parameter
efficient method for fine-tuning LLMs (PEFT). It works by decomposing the
update matrix $\delta W$ into two lower-rank matrices so that instead of tuning
$M*N$ parameters, you only have to tune $M*r+r*N$ parameters where $r<<M$ and
$r<<N$. It appears that the default value for $r=2$ in general.

