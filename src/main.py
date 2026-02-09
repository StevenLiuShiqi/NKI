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
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

from model import NeuronGPTOSSForCausalLM
from gpt_oss.modeling_gpt_oss_orig import NeuronGptOssForCausalLM

# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["XLA_IR_DEBUG"]= "1"
# os.environ["XLA_HLO_DEBUG"]= "1"
# os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
# os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
# os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

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
    parser.add_argument("--run-accuracy-generate", action="store_true",
                        help="Also run get_generate_outputs() after HF adapter generate")
    
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

    # Benchmarking
    parser.add_argument("--benchmark", action="store_true",
                       help="Enable benchmarking mode to measure latency and throughput")
    parser.add_argument("--benchmark-report-path", type=str, default="./benchmark_report.json",
                       help="Path to save benchmark results JSON file")
    parser.add_argument("--num-benchmark-runs", type=int, default=20,
                       help="Number of benchmark iterations for statistical accuracy")

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
        "tp_degree", "batch_size", "max_batch_size", "buckets",
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
        # sequence/bucket sizing
        "seq_len", "max_length", "n_positions", "n_active_tokens", "max_context_length",
        # tokenizer / mask layout
        "padding_side",
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
    # Keep compiled/runtime batch dimensions aligned to avoid sequential fallback in ModelWrapper.
    if "batch_size" in config_kwargs and config_kwargs["batch_size"] is not None:
        neuron_kwargs["batch_size"] = config_kwargs["batch_size"]
        neuron_kwargs.setdefault("max_batch_size", config_kwargs["batch_size"])
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


def run_benchmarking(model, generation_config, num_runs, benchmark_report_path):
    """
    Run comprehensive benchmarking on the model.

    Args:
        model: Neuron model instance (NeuronGptOssForCausalLM)
        generation_config: GenerationConfig object with sampling parameters
        num_runs: Number of benchmark iterations
        benchmark_report_path: Path to save JSON results

    Returns:
        dict: Benchmark results containing latency percentiles and throughput
    """
    print(f"\nStarting benchmark with {num_runs} runs...")
    print(f"Results will be saved to: {benchmark_report_path}")

    results = benchmark_sampling(
        model=model,
        draft_model=None,
        generation_config=generation_config,
        target="all",
        image=False,
        num_runs=num_runs,
        benchmark_report_path=benchmark_report_path,
    )

    print("\nBenchmark Summary:")
    if "e2e_model" in results:
        e2e = results["e2e_model"]
        print(f"  End-to-End Latency (avg): {e2e['latency_ms_avg']:.2f} ms")
        print(f"  End-to-End Latency (P50): {e2e['latency_ms_p50']:.2f} ms")
        print(f"  End-to-End Latency (P99): {e2e['latency_ms_p99']:.2f} ms")
        print(f"  Throughput: {e2e['throughput']:.2f} tokens/sec")

    return results


def main():
    args = parse_args()

    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]
    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    args.tol_map = "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"

    model, tokenizer, generation_config = prepare_inference(NeuronGptOssForCausalLM, args)
    
    print("Compiled!")
    
    batch_size = model.neuron_config.batch_size
    sampling_params = prepare_sampling_params(
        batch_size=batch_size,
        top_k=[args.top_k] * batch_size,
        top_p=[args.top_p] * batch_size,
        temperature=[args.temperature] * batch_size,
    )
    print(f"Prompts: {args.prompts}")
    inputs = tokenizer(args.prompts, padding=True, return_tensors="pt")
    print(f"Input ids: {inputs}")
    if inputs.input_ids.shape[0] > model.neuron_config.batch_size:
        raise RuntimeError(
            f"Runtime batch ({inputs.input_ids.shape[0]}) exceeds compiled batch "
            f"({model.neuron_config.batch_size}). This can cause KV-cache contamination. "
            "Recompile with matching NeuronConfig batch_size/max_batch_size."
        )
    
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
    
    if args.run_accuracy_generate:
        run_generation(
            model,
            tokenizer,
            args.prompts,
            generation_config
        )

    # Benchmarking integration (optional, runs when --benchmark flag is set)
    if args.benchmark:
        run_benchmarking(
            model=model,
            generation_config=generation_config,
            num_runs=args.num_benchmark_runs,
            benchmark_report_path=args.benchmark_report_path
        )

if __name__ == "__main__":
    main()
