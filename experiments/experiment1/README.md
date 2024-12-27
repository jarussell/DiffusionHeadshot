# Experiment 1


The LoRA adapter was trained on the 3 family members. I had a few hypotheses about what to expect.

## Hypotheses

H1. The amount of training data would need to grow over what a normal (single-character) LoRA would require.

Specifically, I expected that if we were to apply the inclusion-exclusion
principle, we would need to fill in every intersection with some number of
training samples. This means for my family with Jacob, Ashley, and Colby, we
would need samples that included each individual, each pair, and all three
together. I hypothesized that we don't need samples without all three people
because that's what the foundation model contained.

## Experiment

Run the full LoRA training without tuning any hyperparameters to get a baseline on the original model.
Generate a headshot and Christmas cards like we intended to train for.

Qualitatively evaluate whether the results were good, bad, or neutral.

Results:

Using the prompt "a professional photograph of JRussell with a clean-shaven
head, wearing a suit and tie," we can see the results below.

The headshot captured my teeth and eyebrows but did a poor job with the beard,
hair, and facial outline. It looks like a cousin but not the same person.

Original Image | Generated Headshot
:-------------:|:------------------:
![Original image](https://github.com/jarussell/DiffusionHeadshot/blob/main/orig_headshot.png?raw=true) | ![Headshot of Jacob from experiment 1](https://github.com/jarussell/DiffusionHeadshot/blob/main/experiments/experiment1/headshot.png?raw=true)

This appears to be because it did not realize that hair was important for the
character in the LoRA. Perhaps because some of the training photos included
hats.

The Christmas card results were really bad. They generated family Christmas
cards that looked great but none of my family looked like my family.

The prompt was "A christmas card with adult ARussell, child CRussell, and adult JRussell lit by the warm glow of the fireplace," and the results are below.

![Generated family Christmas card photo](https://github.com/jarussell/DiffusionHeadshot/blob/main/experiments/experiment1/fam_xmas.png?raw=true)

Limitations of my GPU were apparent. Because I did not have the 24GiB of memory
to train a LoRA all on the GPU, I trained a quantized LoRA with a quantized
model so that it could all fit in memory. This led to poor prompt adherence,
just like indicated in the huggingface repo. Since I have 2 cards, I could try
to run a slightly larger model and distribute the load across both cards.

## Next steps

This experiment used the default parameters for training a LoRA adapter from
the huggingface research examples repo. These parameters were intended to train
a style LoRA and should be adjusted for a character LoRA instead. It would be
nice to try a few different experiments: 1) just train a single-character LoRA.
