def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",trust_remote_code=True).to("cuda")
    model = model.eval()
    return model, tokenizer
