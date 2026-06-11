import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def compress_model_4bit(model_id, save_dir=None):
    """
    Load and compress a model using 4-bit quantization (bitsandbytes).

    Args:
        model_id (str): Hugging Face model name or local path.
        save_dir (str, optional): Directory to save the quantized model and tokenizer.

    Returns:
        model: Quantized model ready for inference.
        tokenizer: Associated tokenizer.
    """
    # 4-bit configuration for maximum memory savings
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   # faster compute
        bnb_4bit_use_double_quant=True,        # extra memory saving
        bnb_4bit_quant_type="nf4"              # better accuracy than fp4
    )

    # Load tokenizer (no quantization needed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",                     # automatically uses GPU if available
        trust_remote_code=True                 # for some custom models
    )

    if save_dir:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Quantized model saved to {save_dir}")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = compress_model_4bit("Qwen/Qwen2.5-1.5B-Instruct", save_dir="./qwen_4bit")

    input_text = "Explain quantum computing in one sentence."
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))