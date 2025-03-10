from modelling_gemma import PaligemmaForConditionalGeneration, PaligemmaConfig 
from transformers import AutoTokenizer 
import json 
import glop 
from safetensors import safe_open 
from typing import Tuple 
import os 

def load_hf_model(model_path: str, device: str) -> Tuple[PaligemmaForConditionalGeneration, AutoTokenizer]:
    # Load The tokenizer 
    tokenizer=AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"
    