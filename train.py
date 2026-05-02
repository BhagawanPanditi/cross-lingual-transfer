import os, json, csv, math, pickle, argparse, logging
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup, set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",            type=str,   required=True)
parser.add_argument("--data_dir",              type=str,   required=True)
parser.add_argument("--val_file",              type=str,   required=True)
parser.add_argument("--subspace_path",         type=str,   required=True)
parser.add_argument("--output_dir",            type=str,   required=True)
parser.add_argument("--resume_from_checkpoint",type=str,   default=None, help="Path to checkpoint directory to resume from")
parser.add_argument("--lora_rank",             type=int,   default=64)
parser.add_argument("--lora_alpha",            type=int,   default=16)
parser.add_argument("--lora_dropout",          type=float, default=0.1)
parser.add_argument("--reg_lambda",            type=float, default=1.0)
parser.add_argument("--learning_rate",         type=float, default=1e-4)
parser.add_argument("--warmup_ratio",          type=float, default=0.03)
parser.add_argument("--per_device_batch_size", type=int,   default=2)
parser.add_argument("--grad_accum_steps",      type=int,   default=8)
parser.add_argument("--max_seq_len",           type=int,   default=1024)
parser.add_argument("--num_epochs",            type=int,   default=20)
parser.add_argument("--eval_steps",            type=int,   default=150)
parser.add_argument("--patience",              type=int,   default=5)
parser.add_argument("--seed",                  type=int,   default=42)
parser.add_argument("--gradient_mask_path",    type=str,   default=None, help="Optional path to gradient masks for LoRA layers")
args = parser.parse_args()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

torch.use_deterministic_algorithms(True)

g = torch.Generator()
g.manual_seed(args.seed)


logger.info("=" * 60)
logger.info("Training Configuration:")
for k, v in vars(args).items():
    logger.info(f"  {k:<30} = {v}")
logger.info(f"  {'device':<30} = {device}")
logger.info("=" * 60)

# ── tokenizer ─────────────────────────────────────────────────────────────────
logger.info(f"Loading tokenizer from {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

with open("base_chat_template.jinja") as f:
    tokenizer.chat_template = f.read()
logger.info("Chat template loaded from base_chat_template.jinja")

if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ── dataset ───────────────────────────────────────────────────────────────────
class MathDataset(Dataset):
    def __init__(self, samples, pos_to_last_idx=None, pos_to_stored_input_ids=None):
        self.samples = samples
        self.pos_to_last_idx = pos_to_last_idx
        self.pos_to_stored_input_ids = pos_to_stored_input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        pos = s.get("_pos", -1)

        stored_last_idx  = self.pos_to_last_idx.get(int(pos), -1)         if self.pos_to_last_idx          else -1
        stored_input_ids = self.pos_to_stored_input_ids.get(int(pos), torch.tensor([])) if self.pos_to_stored_input_ids else torch.tensor([])

        input_ids = []
        labels    = []
        messages  = s["messages"]

        if tokenizer.bos_token_id is not None:
            input_ids += [tokenizer.bos_token_id]
            labels    += [-100]

        for i, msg in enumerate(messages):
            role    = msg["role"]
            content = msg["content"]

            if role == "user":
                text = f"User: {content}\n\n"
                toks = tokenizer.encode(text, add_special_tokens=False)
                input_ids += toks
                labels    += [-100] * len(toks)

            elif role == "assistant":
                prefix = "Assistant: "
                suffix = tokenizer.eos_token
                if i < len(messages) - 1:
                    suffix += "\n\n"

                text          = f"{prefix}{content}{suffix}"
                combined_toks = tokenizer.encode(text, add_special_tokens=False)

                prefix_no_space_toks = tokenizer.encode("Assistant:", add_special_tokens=False)
                mask_len = len(prefix_no_space_toks)

                input_ids += combined_toks
                labels    += [-100] * mask_len + combined_toks[mask_len:]

        input_ids = input_ids[:args.max_seq_len]
        labels    = labels[:args.max_seq_len]

        return pos, input_ids, labels, stored_last_idx, stored_input_ids


def collate_fn(batch):
    pos_ids, all_input_ids, all_labels, stored_last_idxs, all_stored_input_ids = zip(*batch)

    max_len = max(len(x) for x in all_input_ids)
    pad_id  = tokenizer.pad_token_id

    input_ids_padded, labels_padded, attention_masks = [], [], []
    for ids, labs in zip(all_input_ids, all_labels):
        pad_len = max_len - len(ids)
        input_ids_padded.append(ids + [pad_id] * pad_len)
        labels_padded.append(labs + [-100] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    max_stored_len = max((len(x) for x in all_stored_input_ids), default=0)
    stored_input_ids_padded = []
    for s_ids in all_stored_input_ids:
        s_ids_list = s_ids.tolist() if isinstance(s_ids, torch.Tensor) else list(s_ids)
        pad_len = max_stored_len - len(s_ids_list)
        stored_input_ids_padded.append(s_ids_list + [pad_id] * pad_len)

    enc = {
        "input_ids":         torch.tensor(input_ids_padded,         dtype=torch.long),
        "attention_mask":    torch.tensor(attention_masks,           dtype=torch.long),
        "labels":            torch.tensor(labels_padded,             dtype=torch.long),
        "stored_last_idx":   torch.tensor(stored_last_idxs,         dtype=torch.long),
        "stored_input_ids":  torch.tensor(stored_input_ids_padded,  dtype=torch.long),
    }
    return list(pos_ids), enc


jsonl_path = f"{args.data_dir}/math_train_modified.jsonl"

with open(jsonl_path) as f:
    train_samples = [json.loads(l) for l in f]

logger.info(f"Total training samples loaded: {len(train_samples)}")

repr_path = f"{args.data_dir}/math_base_representations.pt"
logger.info(f"Loading base representations from {repr_path}")

repr_shard = torch.load(repr_path, map_location="cpu", weights_only=False)

base_vecs          = repr_shard["vecs"]
shard_ids          = repr_shard["ids"]
shard_last_indices = repr_shard["last_indices"] 
all_input_ids_repr = repr_shard["input_ids"]

pos_to_row      = {int(shard_ids[i]): i           for i in range(len(shard_ids))}
pos_to_last_idx = {int(shard_ids[i]): int(shard_last_indices[i]) for i in range(len(shard_ids))}
pos_to_stored_input_ids = {int(shard_ids[i]): all_input_ids_repr[i] for i in range(len(shard_ids))}

logger.info(f"Base vecs shape: {base_vecs.shape}")

# ── validation data ───────────────────────────────────────────────────────────
logger.info(f"Loading validation data from {args.val_file}")
with open(args.val_file) as f:
    val_samples = [json.loads(l) for l in f]
logger.info(f"Loaded {len(val_samples)} validation samples")

val_loader = DataLoader(
    MathDataset(val_samples),
    batch_size=args.per_device_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
    generator=g
)

# ── language subspace ─────────────────────────────────────────────────────────
logger.info(f"Loading language subspace from {args.subspace_path}")
with open(args.subspace_path, "rb") as f:
    lang_space = pickle.load(f)
lang_space = lang_space[-1].to(device).to(torch.bfloat16)
logger.info(f"lang_space shape: {lang_space.shape}")

# ── gradient masking ──────────────────────────────────────────────────────────
def register_lora_gradient_masks(model, masks: Dict[str, List[torch.Tensor]], device: str = "cuda"):
    """
    Register backward hooks on LoRA layers to mask gradients.
    """
    handles = []
    
    def make_grad_hook(mask):
        mask_gpu = mask.to(device)
        def hook(grad):
            return grad * mask_gpu.to(grad.dtype)
        return hook
    
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
            layers = base_model.model.model.layers
        elif hasattr(base_model, 'model'):
            layers = base_model.model.layers
        else:
            raise ValueError("Could not find model layers in PEFT structure")
    else:
        layers = model.model.layers
    
    num_layers = len(layers)
    
    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        
        # gate_proj
        if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'lora_B'):
            for adapter_name, lora_B_module in mlp.gate_proj.lora_B.items():
                if lora_B_module.weight.requires_grad:
                    mask = masks['gate_proj.lora_B'][layer_idx]
                    handles.append(lora_B_module.weight.register_hook(make_grad_hook(mask)))
        
        # up_proj  
        if hasattr(mlp, 'up_proj') and hasattr(mlp.up_proj, 'lora_B'):
            for adapter_name, lora_B_module in mlp.up_proj.lora_B.items():
                if lora_B_module.weight.requires_grad:
                    mask = masks['up_proj.lora_B'][layer_idx]
                    handles.append(lora_B_module.weight.register_hook(make_grad_hook(mask)))
        
        # down_proj
        if hasattr(mlp, 'down_proj') and hasattr(mlp.down_proj, 'lora_A'):
            for adapter_name, lora_A_module in mlp.down_proj.lora_A.items():
                if lora_A_module.weight.requires_grad:
                    mask = masks['down_proj.lora_A'][layer_idx]
                    handles.append(lora_A_module.weight.register_hook(make_grad_hook(mask)))
    
    logger.info(f"Registered {len(handles)} LoRA gradient mask hooks across {num_layers} layers")
    return handles

# ── model + LoRA ──────────────────────────────────────────────────────────────
logger.info(f"Loading base model from {args.model_path}")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

if args.resume_from_checkpoint:
    logger.info(f"Loading LoRA adapters from checkpoint: {args.resume_from_checkpoint}")
    model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
else:
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

model = model.to(device)

model.print_trainable_parameters()

# ── APPLY OPTIONAL GRADIENT MASKS ──
gradient_mask_handles = []
if args.gradient_mask_path:
    if os.path.exists(args.gradient_mask_path):
        logger.info(f"Applying LoRA gradient masks from: {args.gradient_mask_path}")
        gradient_masks = torch.load(args.gradient_mask_path, map_location=device)
        gradient_mask_handles = register_lora_gradient_masks(
            model, 
            gradient_masks, 
            device=str(device)
        )
    else:
        logger.warning(f"Gradient mask path '{args.gradient_mask_path}' does not exist! Proceeding without masking.")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── optimizer + scheduler ─────────────────────────────────────────────────────
N_total         = len(train_samples)
steps_per_epoch = math.ceil(N_total / args.per_device_batch_size / args.grad_accum_steps)
total_steps     = steps_per_epoch * args.num_epochs
warmup_steps    = int(total_steps * args.warmup_ratio)

logger.info(f"Steps per epoch : {steps_per_epoch}")
logger.info(f"Total steps     : {total_steps}")
logger.info(f"Warmup steps    : {warmup_steps}")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate,
    weight_decay=0.0,
)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Resume State Tracking ─────────────────────────────────────────────────────
start_epoch        = 0
global_step        = 0
best_val_loss      = float("inf")
early_stop_counter = 0
batches_to_skip    = 0

if args.resume_from_checkpoint:
    state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
    if os.path.exists(state_path):
        logger.info(f"Loading optimizer, scheduler, and RNG states from {state_path}")
        ckpt = torch.load(state_path, map_location="cpu")
        
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        
        global_step        = ckpt["global_step"]
        best_val_loss      = ckpt.get("best_val_loss", float("inf"))
        early_stop_counter = ckpt.get("early_stop_counter", 0)

        if "rng_state_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_state_torch"])
        if "rng_state_cuda" in ckpt and ckpt["rng_state_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])
            
        # Calculate exactly where we are in the dataset
        start_epoch     = global_step // steps_per_epoch
        batches_to_skip = (global_step % steps_per_epoch) * args.grad_accum_steps
        
        logger.info(f"Successfully resumed at global_step {global_step} (Epoch {start_epoch}).")
        if batches_to_skip > 0:
            logger.info(f"Will skip {batches_to_skip} micro-batches to align dataloader.")
    else:
        logger.warning(f"No training_state.pt found in {args.resume_from_checkpoint}. Starting step tracking from scratch.")

# ── CSV logger ────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

# Append to CSV if resuming, otherwise create new
mode = "a" if args.resume_from_checkpoint else "w"
train_csv    = open(f"{args.output_dir}/train_log.csv", mode, newline="")
val_csv      = open(f"{args.output_dir}/val_log.csv",   mode, newline="")
train_writer = csv.writer(train_csv)
val_writer   = csv.writer(val_csv)

if not args.resume_from_checkpoint:
    train_writer.writerow(["step","epoch","task_loss","reg_loss","total_loss","lr"])
    val_writer.writerow(["step","epoch","val_loss","best_val_loss","early_stop_counter"])
    
logger.info(f"CSV logs will be saved to {args.output_dir}/")

# ── helpers ───────────────────────────────────────────────────────────────────
def get_last_token_hidden(out, attention_mask):
    last_hidden  = out.hidden_states[-1]
    last_indices = attention_mask.sum(dim=1) - 1
    B = last_hidden.size(0)
    return last_hidden[
        torch.arange(B, device=last_hidden.device), last_indices
    ].to(torch.bfloat16)

def projection(vec, space):
    space = space / torch.linalg.norm(space, dim=1, keepdim=True).to(vec.dtype)
    return torch.matmul(torch.matmul(vec, space.T), space)

def reg_loss_fn(h_current, h_base):
    curr_proj = projection(h_current, lang_space)
    base_proj = projection(h_base,    lang_space)
    return nn.MSELoss()(curr_proj, base_proj)

def evaluate():
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    val_bar = tqdm(val_loader, desc="  Validating", leave=False)
    with torch.no_grad():
        for _, enc in val_bar:
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels         = enc["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += out.loss.item()
            n_batches  += 1
            val_bar.set_postfix({"batch_loss": f"{out.loss.item():.4f}"})

    model.train()
    return total_loss / n_batches

# ── training loop ─────────────────────────────────────────────────────────────
should_stop = False

logger.info("Starting training...")

for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Epochs", position=0):
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
    logger.info(f"{'='*50}")

    train_loader = DataLoader(
        MathDataset(
            train_samples,
            pos_to_last_idx=pos_to_last_idx,
            pos_to_stored_input_ids=pos_to_stored_input_ids,
        ),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g
    )
    model.train()
    optimizer.zero_grad()  # clear at start of each epoch

    running_task = 0.0
    running_reg  = 0.0
    running_tot  = 0.0
    micro_step   = 0       # counts individual forward passes within an accum window
    
    # Handle batch skipping if resuming mid-epoch
    skip_for_this_epoch = batches_to_skip if epoch == start_epoch else 0

    train_bar = tqdm(
        train_loader,
        desc=f"  Epoch {epoch+1} train",
        position=1,
        leave=False,
    )

    for pos_ids, enc in train_bar:
        # Fast-forwarding dataloader
        if skip_for_this_epoch > 0:
            skip_for_this_epoch -= 1
            micro_step += 1
            continue

        micro_step += 1

        input_ids       = enc["input_ids"].to(device)
        attention_mask  = enc["attention_mask"].to(device)
        labels          = enc["labels"].to(device)

        # ── forward ───────────────────────────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            task_loss = out.loss

            h_current = get_last_token_hidden(out, attention_mask)
            rows      = [pos_to_row[int(p)] for p in pos_ids]
            h_base    = base_vecs[rows].to(device).to(torch.bfloat16)
            reg_loss  = reg_loss_fn(h_current, h_base)

            # Scale loss by grad_accum_steps so gradients accumulate correctly
            total_loss = (task_loss + args.reg_lambda * reg_loss) / args.grad_accum_steps

        # ── backward ──────────────────────────────────────────────────────────
        total_loss.backward()

        # ── optimizer step every grad_accum_steps micro-batches ───────────────
        sync_gradients = (micro_step % args.grad_accum_steps == 0)

        if sync_gradients:
            global_step += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Unscale losses for logging (undo the /grad_accum_steps)
            t   = task_loss.item()
            r   = reg_loss.item()
            tot = (task_loss + args.reg_lambda * reg_loss).item()

            alpha        = 0.1
            running_task = alpha * t   + (1 - alpha) * running_task
            running_reg  = alpha * r   + (1 - alpha) * running_reg
            running_tot  = alpha * tot + (1 - alpha) * running_tot

            lr = scheduler.get_last_lr()[0]

            train_bar.set_postfix({
                "step"  : global_step,
                "task"  : f"{running_task:.4f}",
                "reg"   : f"{running_reg:.4f}",
                "total" : f"{running_tot:.4f}",
                "lr"    : f"{lr:.2e}",
                "val"   : f"{best_val_loss:.4f}",
            })

            train_writer.writerow([
                global_step, epoch,
                round(t,   6),
                round(r,   6),
                round(tot, 6),
                round(lr,  8),
            ])
            train_csv.flush()

            if global_step % 10 == 0:
                logger.info(
                    f"[step {global_step:>5}] epoch={epoch+1} "
                    f"task={t:.4f}  reg={r:.4f}  total={tot:.4f}  lr={lr:.2e}"
                )

            # ── evaluation block ───────────────────────────────────────────────
            if global_step % args.eval_steps == 0:
                logger.info(f"\n--- Running Validation at Step {global_step} ---")
                val_loss = evaluate()

                if val_loss < best_val_loss:
                    best_val_loss      = val_loss
                    early_stop_counter = 0
                    
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    
                    # ── Save Training State for Resumption ──────────────────────
                    ckpt_state = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "early_stop_counter": early_stop_counter,
                        "rng_state_torch": torch.get_rng_state(),
                        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    }
                    torch.save(ckpt_state, os.path.join(args.output_dir, "training_state.pt"))
                    
                    logger.info(f"  ✓ New best val_loss={val_loss:.4f} — model & state saved to {args.output_dir}")
                else:
                    early_stop_counter += 1
                    logger.info(
                        f"  ✗ val_loss={val_loss:.4f} did not improve "
                        f"(best={best_val_loss:.4f}, patience={early_stop_counter}/{args.patience})"
                    )

                val_writer.writerow([global_step, epoch, round(val_loss,6), round(best_val_loss,6), early_stop_counter])
                val_csv.flush()

                if early_stop_counter >= args.patience:
                    logger.info(
                        f"Early stopping triggered at step {global_step} — "
                        f"no improvement for {args.patience} evaluations."
                    )
                    should_stop = True
                    break   # break inner (train_bar) loop

    if should_stop:
        break   # break outer (epoch) loop

train_csv.close()
val_csv.close()
logger.info("Training complete.")
logger.info(f"Best val_loss : {best_val_loss:.4f}")
logger.info(f"Adapters saved: {args.output_dir}")
logger.info("To get the merged model, run merge.py with this output_dir as adapter_path.")
