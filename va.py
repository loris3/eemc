from transformers import AutoModelForCausalLM

model_name = "allenai/OLMo-2-0425-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
for name, module in model.named_modules():
    print(name)