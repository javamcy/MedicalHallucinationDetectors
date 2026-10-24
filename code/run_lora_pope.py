 
#!/usr/bin/env python3
"""Run the faithful LLaVA-LoRA training script on POPE by patching fair_data first."""
import os
import sys

sys.path.insert(0, "/root/hallu_bench/code")
import pope_ds

pope_ds.install()

src = open("/root/hallu_bench/code/llava_lora.py", encoding="utf-8").read()
sys.argv = ["llava_lora.py", "pope_random"]
g = {"__name__": "__main__", "__file__": "/root/hallu_bench/code/llava_lora.py"}
exec(compile(src, "llava_lora.py", "exec"), g)
