from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_DIRECTORY_PATH = '/home/ubuntu/models/gpt-oss-20b'


if __name__ == '__main__':
    # Load everything
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY_PATH)
    generation_config = GenerationConfig.from_pretrained(MODEL_DIRECTORY_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIRECTORY_PATH)

    # Prompt
    prompt = 'I believe the meaning of life is'

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')

    print(inputs)

    # Generate
    output_id_sequences = model.generate(
        **inputs,
        generation_config=generation_config
    )

    print(output_id_sequences)

    # Decode and print
    output_text = [
        tokenizer.decode(output_id_sequence)
        for output_id_sequence
        in output_id_sequences
    ]

    print(output_text)