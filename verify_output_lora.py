"""MS-Swift callback to validate the installed Qwen text adapters on every rank."""

import re

import torch
from swift.callbacks import TrainerCallback, callbacks_map


class VerifyOutputLoRA(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        expected = set()
        for index in range(64):
            attention = "self_attn.o_proj" if index % 4 == 3 else "linear_attn.out_proj"
            expected.update({f"{index}.{attention}", f"{index}.mlp.down_proj"})

        matched = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                match = re.search(r"(?:^|\.)language_model\.layers\.(.+)$", name)
                matched.append(match.group(1) if match else name)
        actual = set(matched)
        if actual != expected or len(matched) != len(expected):
            raise ValueError(f"Unexpected LoRA targets: missing={sorted(expected - actual)}, "
                             f"extra={sorted(actual - expected)}, count={len(matched)}")

        trainable = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters() if p.requires_grad)
        rank = model.peft_config["default"].r
        if trainable != 2162688 * rank:
            raise ValueError(f"Expected {2162688 * rank} trainable parameters, found {trainable}")
        if state.is_world_process_zero:
            print(f"Verified 16 self_attn.o_proj + 48 linear_attn.out_proj + 64 mlp.down_proj; "
                  f"{trainable:,} trainable parameters at rank {rank}.", flush=True)
            print("Matched language_model.layers:", ", ".join(matched), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"rank={args.process_index} cuda={device} "
                  f"peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB "
                  f"peak_reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB", flush=True)


callbacks_map["verify_output_lora"] = VerifyOutputLoRA
