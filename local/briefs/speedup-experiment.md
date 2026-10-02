Get context of the experiment we ran in from /workspace/speculators/local You are an expert Senior MLE with an IQ of over 170 and lots of experience with speculative decoding:

I want you to use agents to do the heavy lifting, be an orchestrator who only delegates and verifies so you never deal with context rot and give me status updates every 5 mins, help me do the following.

We are running an ablation around quantization of spec-decode drafters. Initially we wanted to see if real calibration data is necessary for quantization of drafters

The two drafters we tested were dflash: https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash and dspark: https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark

We quantized them using different schemes such as: - Data Free: - FP8-Block - FP8-Dynamic - Requiring Calibration (were generated with random gaussian data + on regenerated PerfectBlend calibration data) - FP8-static - NVFP4 - NVFP4-GPTQ - NVFP4-GPTQ-IMatrix

I think based on the experiments so far, we can see with acceptance rates that real calibration data is required. (Confirm this by looking at the results)

Now we want to check what kind of speedups can we expect from quantization of drafters

Help me create an experiment around this:

1. What evals should I run? What is speedbench 1k/2k/8k and how does it differ from speedbench qualitative
2. What kind of numbers should I be plotting to show this speed up and clearly see the expected speedup

Talk to me like an expert researcher and help me figure out gaps in my thinking. (I have less time so this experiment should be crisp and include only relevant things we need to run)

This experiment will be run on a different machine, so help me create a file with all context to run this experiment such that I don't have to repeatedly give another codex session instructions.

All the quantized drafters have been uploaded to https://huggingface.co/collections/inference-optimization/quantized-drafters collection

Kindly also figure out the changes we made to vllm (to allow deployment of these drafters) and include them in the doc.

The deployment commands for these drafters should be as follows:

- Dflash:

```
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"dflash","num_speculative_tokens":7,"model":<drafter-hf-stub>}'
```

- Dspark:

```
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"model":<drafter-hf-stub>}'
```

The idea would be to compare the real data quantized drafters with bf16 baseline (dflash baseline: `RedHatAI/Qwen3-8B-speculator.dflash` dspark baseline: `RedHatAI/Qwen3.8-27B-speculator.dspark`)

First interview me relentlessly till we reach shared understanding, and then create the hand-off speed-up experiment blog
