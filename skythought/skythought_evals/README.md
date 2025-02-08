# Skythought-evals: Data Generation and Evaluation Tools

## Requirements 

Make sure you have installed the `skythought-evals` package as outlined in the [README.md](../README.md).

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```

## Generation and Evaluation
The file `-m skythought_evals.inference_and_check` provides convenient methods for generating sequences (e.g., for distillation or benchmark evaluation) and checking whether the generated solutions are correct (e.g., for reject sampling or benchmark evaluation).

### Benchmark Evaluation
We provide a wrapper script `eval.py` to conveniently run reasoning benchmarks. This script can be used to launch evaluations for multiple benchmarks, then aggregate and log the accuracy for all benchmarks.  To see the full list of supported args and valid arguments, run `python -m skythought_evals.eval --help`

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 

**NOTE**: For reproducing `Sky-T1-32B-Preview` results on `AIME` and `GPQADiamond` dataset, pass in temperatures as `0.7`. 

```shell
python -m skythought_evals.eval --model NovaSky-AI/Sky-T1-32B-Preview --evals=aime,gpqa_diamond --tp=8 --output_file=results.txt --temperatures 0.7 
```

#### Example Usage
```shell
python -m skythought_evals.eval --model Qwen/QwQ-32B-Preview --evals=aime,math500,gpqa_diamond --tp=8 --output_file=results.txt
```
    
Example result: `{"AIME": <aime_accuracy>, "MATH500": <math500_accuracy>, "GPQADiamond": <gpqa_diamond_accuracy>}` 

#### Best-of-N Evaluation

While we are actively working on a better CLI interface, you can use `-m skythought_evals.inference_and_check` for Best-of-N evaluation. 

```bash
python -m skythought_evals.inference_and_check --task math500 --model Qwen/Qwen2-7B-Instruct --tp 4 --max_tokens 4096 --split test --result-dir ./ --inference --temperatures 0.7 --n 64
python -m skythought_evals.inference_and_check --task math500 --model Qwen/Qwen2-7B-Instruct --tp 4 --max_tokens 4096 --split test --result-dir ./ --check --temperatures 0.7 --n 8
```

### Distill and Reject Sampling
Currently we support distill and reject sampling from various self-hosted models for NUMINA, APPS, and TACO datasets. For NUMINA, the source can be one from `[amc_aime, math, olympiads]`.
#### Example Usage

```shell
python -m skythought_evals.inference_and_check --task apps --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --difficulty all --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task taco --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --difficulty MEDIUM --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task taco --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --difficulty all --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source math --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 4 --math-difficulty-upper-bound 9

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source amc_aime --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 1 --math-difficulty-upper-bound 9

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --end 20000--source olympiads --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 9 --math-difficulty-upper-bound 9
```

### Evalution on Intel Gaudi

#### use tgi/vllm backend server

##### setup backend server

the command for tgi gaudi

```

docker run -d -p 8085:80 -v /mnt/disk2/lkk:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PREFILL_BATCH_BUCKET_SIZE=1 -e BATCH_BUCKET_SIZE=8 --cap-add=sys_nice --ipc=host -e https_proxy=$https_proxy -e http_proxy=$http_proxy ghcr.io/huggingface/tgi-gaudi:latest --model-id Qwen/Qwen2.5-32B-Instruct --max-input-tokens 8192 --max-total-tokens 32768 --sharded true --num-shard 4 --max-batch-total-tokens 65536 --max-batch-prefill-tokens 8192

```

##### do evaluation

```
python -m skythought_evals.inference_and_check --task aime --model Qwen/Qwen2.5-32B-Instruct --tp 8 --result-dir ./ --temperatures 0.7 --hpu --use_openai --endpoint http://localhost:8085/v1 --max_tokens 16384


# note: If --model don't in the built-in model list, you should set --system-prompt-template to use system template

python -m skythought_evals.inference_and_check --task aime --model /lkk/Sky-T1-32B-Preview--tp 8 --result-dir ./ --temperatures 0.7 --hpu --use_openai --endpoint http://localhost:8085/v1 --max_tokens 16384 --system-prompt-template skythought

```


#### Use vllm setup model

```
python -m skythought_evals.inference_and_check --task aime --model /lkk/Sky-T1-32B-Preview --tp 8 --result-dir ./ --temperatures 0.7 --hpu --system-prompt-template skythought

# note: If --model don't in the built-in model list, you should set --system-prompt-template to use system template

```
