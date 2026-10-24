 
"""VASE (full) on MedGemma-4b-it: monkey-patch the model layer of vase_reproduction."""
import os, sys, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/root/hallu_bench/code")
sys.stdout.reconfigure(line_buffering=True)
from PIL import Image
import numpy as np
import vase_reproduction as VR

MG = "/cloud/hf_models/medgemma-4b-it"
_proc = None
_model = None

def load_all():
    global _proc, _model
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _proc = AutoProcessor.from_pretrained(MG)
    _model = AutoModelForImageTextToText.from_pretrained(
        MG, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    print("[medgemma] loaded", flush=True)
    # return placeholders matching the (tokenizer, model, image_processor, dtok, dmodel) contract
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dsnap = os.environ.get("DEBERTA_MODEL", "microsoft/deberta-v2-xlarge-mnli")
    dtok = AutoTokenizer.from_pretrained(dsnap)
    dmodel = AutoModelForSequenceClassification.from_pretrained(
        dsnap, torch_dtype=torch.float16).to(VR.DEV).eval()
    print("[medgemma] deberta loaded", flush=True)
    return (_proc.tokenizer, _model, _proc, dtok, dmodel)

def make_inputs(tokenizer, image_processor, image, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = image_processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = image_processor(text=[prompt], images=[image], return_tensors="pt")
    if "pixel_values" in enc and enc["pixel_values"].dim() == 5:
        enc["pixel_values"] = enc["pixel_values"][0]
    ids = enc["input_ids"]
    img = enc["pixel_values"]
    attn = enc["attention_mask"]
    return ids, img, attn

def generate_batch(model, tokenizer, input_ids, img_t, attn, n, temperature):
    prm = next(model.parameters())
    dev, dt = prm.device, prm.dtype
    b = input_ids.shape[0]
    ids = input_ids.repeat(n, 1).to(dev)
    a = attn.repeat(n, 1).to(dev)
    img = img_t.repeat(n, *([1] * (img_t.dim() - 1))).to(dev).to(dt)
    with torch.no_grad():
        out = model.generate(ids, pixel_values=img, attention_mask=a,
                             max_new_tokens=VR.MAX_NEW,
                             do_sample=(temperature > 0.05), temperature=max(temperature, 0.05),
                             num_beams=1, output_scores=True,
                             return_dict_in_generate=True, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen = out.sequences[:, ids.shape[1]:]
    scores = out.scores  # tuple of [n, V]
    ans = tokenizer.batch_decode(gen, skip_special_tokens=True)
    ans = [x.strip() for x in ans]
    return ans, gen, scores

VR.load_all = load_all
VR.make_inputs = make_inputs
VR.generate_batch = generate_batch

if __name__ == "__main__":
    ds = sys.argv[1]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    VR.main(ds, max_n=mx)
