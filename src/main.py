import argparse
import time
import copy
import sys
import os

# Add parent directory to path to find gpt_oss package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from model import NeuronGPTOSSForCausalLM
from src.gpt_oss.modeling_gpt_oss_orig import NeuronGptOssForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/models/gpt-oss-20b/")
    parser.add_argument("--compiled-model-path", type=str,
                        default="/home/ubuntu/traced_model/")
    
    # Generation
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--global-topk", type=int)
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=2)
    
    # Basic config
    # parser.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    # parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--padding-side", type=str)
    parser.add_argument("--seq-len", type=int, default=64)
    # parser.add_argument("--n-active-tokens", type=int)
    # parser.add_argument("--n-positions", type=int)
    # parser.add_argument("--max-context-length", type=int)
    # parser.add_argument("--max-new-tokens", type=int)
    # parser.add_argument("--max-length", type=int)
    # parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    # parser.add_argument("--output-logits", action="store_true")
    # parser.add_argument("--vocab-parallel", action="store_true")

    # # Attention
    parser.add_argument("--fused-qkv", dest="fused_qkv", type=bool, default=False)
    # parser.add_argument("--sequence-parallel-enabled", action="store_true")
    # parser.add_argument("--flash-decoding-enabled", action="store_true")

    # # On device sampling
    # parser.add_argument("--on-device-sampling", action="store_true")

    # # Bucketing
    # parser.add_argument("--enable-bucketing", type=bool, default=True)
    # parser.add_argument("--bucket-n-active-tokens", action="store_true")
    # parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    # parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # # Parallelism
    parser.add_argument("--tp-degree", dest="tp_degree", type=int, default=8)
    
    # Padding
    parser.add_argument("--padded-hidden-size", dest="padded_hidden_size", type=int, default=2944,
                        help="Padded hidden size (must be >= hidden_size and divisible by 128 for MoE kernel)")

    # # Kernels
    # parser.add_argument("--qkv-kernel-enabled", action="store_true")
    # parser.add_argument("--attn-kernel-enabled", action="store_true")
    # parser.add_argument("--mlp-kernel-enabled", action="store_true")
    # parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    # parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    # parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    # parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    return parser.parse_args()



def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def prepare_inference(model_cls, args):
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    
     # Keys that belong to Neuron*Config
    neuron_keys = {
        # common MoE/Neuron runtime knobs; 
        "tp_degree", "max_batch_size", "buckets",
        "torch_dtype", "on_device_sampling_config",
        # kernel toggles 
        "qkv_kernel_enabled", "attn_kernel_enabled", "mlp_kernel_enabled",
        "quantized_mlp_kernel_enabled", "rmsnorm_quantize_kernel_enabled",
        "quantized_kernel_lower_bound", "mlp_kernel_fuse_residual_add",
        "fused_qkv", "sequence_parallel_enabled", "flash_decoding_enabled",
        "enable_bucketing", "bucket_n_active_tokens",
        "context_encoding_buckets", "token_generation_buckets",
        # padding
        "padded_hidden_size", "padded_intermediate_size",
    }

    # Keys that *do not* belong to NeuronConfig 
    generation_keys = {
        "prompts", "top_k", "top_p", "temperature", "do_sample",
        "dynamic", "pad_token_id", "global_topk"
    }
    path_keys = {"model_path", "compiled_model_path"}
    driver_only = {"seq_len", "batch_size", "max_length", "tol_map", "padding_side"}

    # Build kwargs for each consumer
    config_kwargs["original_hidden_size"] = 2880
    config_kwargs["is_mxfp4_compute"] = False
    neuron_kwargs = {k: config_kwargs[k] for k in neuron_keys if k in config_kwargs and config_kwargs[k] is not None}
    # (Do NOT pass generation/path args into NeuronConfig)

    # if args.on_device_sampling:
    #     config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

    neuron_config = model_cls.get_neuron_config_cls()(**neuron_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    # Compile and save model.
    compiling_start_time = time.monotonic()
    print("\nCompiling and saving model...")
    model.compile(args.compiled_model_path, debug=False)

    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)
    neuron_config.pad_token_id = tokenizer.pad_token_id

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    remaining_kwargs = generation_config.update(**generation_config_kwargs)

    # add any remaining ones (this can happen when the model generation config is missing some entries)
    for k, v in remaining_kwargs.items():
        generation_config.__dict__[k] = v

    return model, tokenizer, generation_config

def run_generation(model, tokenizer, prompts, generation_config):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    outputs, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=model.neuron_config.max_length,
    )

    print(outputs)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def main():
    args = parse_args()

    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]
    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    args.tol_map = "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"

    model, tokenizer, generation_config = prepare_inference(NeuronGptOssForCausalLM, args)
    
    print("Compiled!")
    
    sampling_params = prepare_sampling_params(batch_size=model.neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {args.prompts}")
    inputs = tokenizer(args.prompts, padding=True, return_tensors="pt")
    print(f"Input ids: {inputs}")
    
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
    
    run_generation(
        model,
        tokenizer,
        args.prompts,
        generation_config
    )
    
if __name__ == "__main__":
    main()