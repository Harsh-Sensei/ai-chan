from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

class Mistral:
    def __init__(self) -> None:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", quantization_config=quantization_config,device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_response(self, msg):
        encodeds = self.tokenizer.apply_chat_template(msg, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])
        return decoded[0][:-4]
