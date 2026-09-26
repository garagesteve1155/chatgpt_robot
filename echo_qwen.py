#!/usr/bin/env python3
"""Local Qwen3-14B cognition engine used by Echo's desktop brain."""
from __future__ import annotations
import gc, json, re, threading
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_QWEN_DIR = r"C:\Users\garag\OneDrive\Desktop\MME\qwen3_14b"
THINK_MODE_PREFIX = "think harder and then choose a command"
THINKING_BUDGETS = {"low":512,"medium":1024,"high":2048,"very high":4096}

class LocalResponse:
    def __init__(self,text,prompt_tokens,completion_tokens,extra=None):
        self.text=text
        self.prompt_tokens=int(prompt_tokens)
        self.completion_tokens=int(completion_tokens)
        self.extra=dict(extra or {})
    def json(self):
        out={
            "choices":[{"message":{"content":self.text}}],
            "usage":{
                "prompt_tokens":self.prompt_tokens,
                "completion_tokens":self.completion_tokens,
                "total_tokens":self.prompt_tokens+self.completion_tokens,
            },
            "local_qwen":True,
        }
        out.update(self.extra)
        return out

class EchoQwen:
    def __init__(self,model_dir=None):
        self.model_dir=str(model_dir or DEFAULT_QWEN_DIR)
        self.model=None
        self.tokenizer=None
        self._load_lock=threading.Lock()
        self._generation_lock=threading.Lock()
        self.last_stats={}

    def load(self):
        with self._load_lock:
            if self.model is not None:
                return self.tokenizer,self.model
            if not torch.cuda.is_available():
                raise RuntimeError("Echo's Qwen3 brain requires an NVIDIA CUDA GPU on the desktop.")
            p=Path(self.model_dir)
            if not p.exists():
                raise RuntimeError(f"Local Qwen3 model folder not found: {p}")
            print("Loading Echo local Qwen3-14B in 4-bit NF4...",flush=True)
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            self.tokenizer=AutoTokenizer.from_pretrained(p,use_fast=False)
            self.tokenizer.pad_token=self.tokenizer.pad_token or self.tokenizer.eos_token
            offload=Path(__file__).resolve().parent/"offload_echo_qwen"
            offload.mkdir(exist_ok=True)
            self.model=AutoModelForCausalLM.from_pretrained(
                p,
                device_map="auto",
                quantization_config=quant,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                offload_folder=str(offload),
                offload_state_dict=True,
            )
            self.model.eval()
            print("Qwen device map:",getattr(self.model,"hf_device_map",None),flush=True)
            print("Echo local Qwen3-14B ready.",flush=True)
            return self.tokenizer,self.model

    @staticmethod
    def _device(model):
        try:return model.device
        except Exception:
            try:return next(model.parameters()).device
            except Exception:return torch.device("cpu")

    @staticmethod
    def _normalize_messages(messages):
        out=[]
        for m in messages:
            content=m.get("content","")
            if isinstance(content,list):
                content="\n".join(
                    str(x.get("text") or "")
                    for x in content
                    if isinstance(x,dict) and x.get("type")=="text"
                )
            out.append({"role":str(m.get("role","user")),"content":str(content)})
        return out

    def _render(self,messages,enable_thinking):
        tok,_=self.load()
        messages=self._normalize_messages(messages)
        if getattr(tok,"apply_chat_template",None) and getattr(tok,"chat_template",None):
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )+"<|im_start|>assistant\n"

    @staticmethod
    def first_json_object(text):
        if not isinstance(text,str):return None
        text=re.sub(r"^```[a-zA-Z]*\s*|\s*```$","",text.strip())
        start=text.find("{")
        if start<0:return None
        depth=0; in_string=False; escaped=False
        for i in range(start,len(text)):
            ch=text[i]
            if in_string:
                if escaped:escaped=False
                elif ch=="\\":escaped=True
                elif ch=='"':in_string=False
                continue
            if ch=='"':in_string=True
            elif ch=="{":depth+=1
            elif ch=="}":
                depth-=1
                if depth==0:
                    try:v=json.loads(text[start:i+1])
                    except json.JSONDecodeError:return None
                    return v if isinstance(v,dict) else None
        return None

    @staticmethod
    def _cleanup():
        if torch.cuda.is_available():
            try:torch.cuda.empty_cache()
            except Exception:pass
        gc.collect()

    def _generate_unlocked(self,messages,*,max_new_tokens=900,enable_thinking=False,thinking_budget=2048,temperature=None,top_p=None):
        tok,mdl=self.load()
        prompt=self._render(messages,enable_thinking)
        inputs=first_out=second_out=continuation=forced=None
        try:
            inputs=tok(prompt,return_tensors="pt",truncation=False).to(self._device(mdl))
            input_len=int(inputs["input_ids"].shape[1])
            if enable_thinking:
                with torch.inference_mode():
                    first_out=mdl.generate(
                        **inputs,
                        max_new_tokens=max(1,int(thinking_budget)),
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        repetition_penalty=1.04,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.eos_token_id,
                    )
                ids=first_out[0][input_len:].tolist()
                think_end=tok.convert_tokens_to_ids("</think>")
                eos=tok.eos_token_id
                finished=(eos in ids) if isinstance(eos,int) else any(x in ids for x in (eos or []))
                if not finished:
                    continuation=first_out
                    if not (isinstance(think_end,int) and think_end in ids):
                        print(f"Qwen thinking budget reached ({thinking_budget}); closing thought and generating command...",flush=True)
                        forced=tok(
                            "\n\nI have used the selected reasoning budget. I will now choose the best actual command.\n</think>\n\n",
                            return_tensors="pt",
                            return_attention_mask=False,
                            add_special_tokens=False,
                        ).input_ids.to(first_out.device)
                        continuation=torch.cat([first_out,forced],dim=-1)
                    mask=torch.ones_like(continuation,dtype=torch.int64)
                    with torch.inference_mode():
                        second_out=mdl.generate(
                            input_ids=continuation,
                            attention_mask=mask,
                            max_new_tokens=max(1,int(max_new_tokens)),
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            repetition_penalty=1.04,
                            eos_token_id=tok.eos_token_id,
                            pad_token_id=tok.eos_token_id,
                        )
                    ids=second_out[0][input_len:].tolist()
                if isinstance(think_end,int) and think_end in ids:
                    rev=ids[::-1].index(think_end)
                    ids=ids[len(ids)-rev:]
            else:
                t=0.7 if temperature is None else float(temperature)
                p=0.8 if top_p is None else float(top_p)
                with torch.inference_mode():
                    first_out=mdl.generate(
                        **inputs,
                        max_new_tokens=max(1,int(max_new_tokens)),
                        do_sample=True,
                        temperature=t,
                        top_p=p,
                        top_k=20,
                        repetition_penalty=1.04,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.eos_token_id,
                    )
                ids=first_out[0][input_len:].tolist()
            text=tok.decode(ids,skip_special_tokens=True).strip()
            resp=LocalResponse(
                text,input_len,len(ids),
                {"thinking_enabled":bool(enable_thinking),"thinking_budget":int(thinking_budget) if enable_thinking else 0}
            )
            self.last_stats=resp.json()["usage"]|resp.extra
            return text,resp
        finally:
            self._cleanup()

    def generate_text(self,messages,**kwargs):
        with self._generation_lock:
            return self._generate_unlocked(messages,**kwargs)

    def generate_json(self,messages,**kwargs):
        messages=list(messages)+[{
            "role":"user",
            "content":"Return one valid JSON object only. Do not use markdown or add text outside the JSON object."
        }]
        with self._generation_lock:
            raw,resp=self._generate_unlocked(messages,**kwargs)
            parsed=self.first_json_object(raw)
            if parsed is not None:return parsed,resp
            repaired,r2=self._generate_unlocked(
                [
                    {"role":"system","content":"Repair the supplied text into one valid JSON object without changing its meaning. Output JSON only."},
                    {"role":"user","content":raw},
                ],
                max_new_tokens=kwargs.get("max_new_tokens",900),
                enable_thinking=False,
                temperature=0.2,
                top_p=0.8,
            )
            parsed=self.first_json_object(repaired)
            if parsed is None:
                raise RuntimeError(f"Qwen failed to return valid JSON. Original={raw!r} Repair={repaired!r}")
            return parsed,LocalResponse(
                repaired,
                resp.prompt_tokens+r2.prompt_tokens,
                resp.completion_tokens+r2.completion_tokens,
                resp.extra,
            )

    @staticmethod
    def _thinking_effort(mode_command):
        text=str(mode_command or "").strip()
        normalized=text.lower().replace("_"," ")
        if not normalized.startswith(THINK_MODE_PREFIX):return None
        requested=text.split("~~",1)[1].strip().lower().replace("_"," ") if "~~" in text else "medium"
        compact=requested.replace(" ","")
        aliases={"veryhigh":"very high","maximum":"very high","max":"very high","normal":"medium"}
        requested=aliases.get(compact,requested)
        return requested if requested in THINKING_BUDGETS else "medium"

    def choose_robot_command(self,messages):
        first,r1=self.generate_json(messages,max_new_tokens=900,enable_thinking=False)
        effort=self._thinking_effort(first.get("mode_command"))
        if effort is None:return first,r1
        budget=THINKING_BUDGETS[effort]
        print(f"Echo chose THINK HARDER at {effort!r} effort ({budget} thinking tokens max).",flush=True)
        second_messages=list(messages)+[
            {"role":"assistant","content":json.dumps(first,ensure_ascii=False)},
            {"role":"user","content":(
                f"You chose 'Think Harder And Then Choose A Command' with {effort} effort. "
                "Now actually think through the current situation and return the real robot command JSON. "
                "Do not choose Think Harder again. The final JSON must be what Echo actually executes."
            )},
        ]
        second,r2=self.generate_json(
            second_messages,
            max_new_tokens=900,
            enable_thinking=True,
            thinking_budget=budget,
        )
        if self._thinking_effort(second.get("mode_command")) is not None:
            second["mode_command"]="false"
        return second,LocalResponse(
            r2.text,
            r1.prompt_tokens+r2.prompt_tokens,
            r1.completion_tokens+r2.completion_tokens,
            {"thinking_enabled":True,"thinking_effort":effort,"thinking_budget":budget},
        )
