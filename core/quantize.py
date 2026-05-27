from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "llama3.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with 4-bit quantization
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True, 
    device_map="auto"
)   