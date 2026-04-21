import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

TOP_RATE = 0.20
FILTER_RATE = 0.95
ACTIVATION_BAR_RATIO = 0.95
WITH_EN = False
LANGUAGES = ["zh", "ja", "bn", "sw", "ru", "de", "es", "fr", "te", "th"]

if WITH_EN:
    LANGUAGES = ["en"] + LANGUAGES
    
n, over_zero = [], []

for lang in LANGUAGES:
    count_dict = torch.load(f"./activation_counts/llama3_activation_counts_{lang}.pt")
    n.append(count_dict['num_tokens'])
    over_zero.append(count_dict['activated_neuron_counts'])

n = torch.tensor(n)
over_zero = torch.stack(over_zero, dim=-1)
num_layers, intermediate_size, lang_num = over_zero.size()


def LAPE():
    activation_probs = over_zero / n # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * FILTER_RATE)).values.item()
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * TOP_RATE)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * ACTIVATION_BAR_RATIO)).values.item()
    lang, indice = torch.where(selected_probs > activation_bar)
    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    return final_indice

def plot_language_neurons(language_specific_neurons,
                             languages,
                             num_layers,
                             save_path="./lang_plots_best.pdf"):
    """
    Plot 10 languages + average in a 2x6 grid.
    """

    l_dict = {
        'bn': 'Bengali', 'de': 'German', 'en': 'English', 'fr': 'French',
        'ru': 'Russian', 'sw': 'Swahili',
        'zh': 'Chinese', 'ja': 'Japanese', 'th': 'Thai',
        'es': 'Spanish', 'te': 'Telugu'
    }

    fig, axs = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)
    axs = axs.flatten()

    # ---- Plot languages ----
    for i, language in enumerate(languages):
        ax = axs[i]

        if i < len(language_specific_neurons):
            layerwise_counts = [tensor.numel() for tensor in language_specific_neurons[i]]
        else:
            layerwise_counts = [0] * num_layers

        ax.bar(range(len(layerwise_counts)),
               layerwise_counts,
               color='#FF9800',
               edgecolor='#5E35B1',
               linewidth=1.2,
               alpha=0.9)

        ax.set_title(l_dict.get(language, language), fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Count')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_xlim(-1, num_layers)

    # ---- Average plot ----
    ax_avg = axs[len(languages)]

    all_counts = []
    for lang_data in language_specific_neurons:
        if lang_data:
            all_counts.append([t.numel() for t in lang_data])
        else:
            all_counts.append([0] * num_layers)

    avg_counts = np.mean(all_counts, axis=0)

    ax_avg.bar(range(len(avg_counts)),
               avg_counts,
               color="#607D8B",
               edgecolor='#263238',
               linewidth=1.2)

    ax_avg.set_title("Average", fontsize=13, fontweight='bold')
    ax_avg.set_xlabel("Layer")
    ax_avg.set_ylabel("Avg Count")
    ax_avg.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_avg.set_xlim(-1, num_layers)

    # ---- Hide extra subplot (12th slot) ----
    for j in range(len(languages) + 1, len(axs)):
        axs[j].axis('off')

    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

language_specific_neurons = LAPE()


plot_language_neurons(
    language_specific_neurons,
    languages=LANGUAGES,
    num_layers=num_layers,
    save_path=f"./language_masks/language_specific_neurons_{TOP_RATE*100:.0f}_{FILTER_RATE*100:.0f}_{ACTIVATION_BAR_RATIO*100:.0f}_{int(WITH_EN)}.pdf"
)

print("LENGTH ", len(language_specific_neurons))
num_languages = len(LANGUAGES)

def build_language_union_mask(language_specific_neurons,
                              hidden_size,
                              intermediate_size,
                              save_path="language_mask.pt"):
    """
    Build complement mask (all neurons NOT in union of language neurons) per layer.
    
    For FFN layers:
        - up_proj and gate_proj have shape (intermediate_size, hidden_size)
        - down_proj has shape (hidden_size, intermediate_size)
    
    We create masks for each projection type:
        - up_proj_mask / gate_proj_mask: (intermediate_size, hidden_size)
        - down_proj_mask: (hidden_size, intermediate_size)

    Mask values:
        1 → allow gradient (complement space - non-language neurons)
        0 → block gradient (language neurons)
    """

    num_languages = len(language_specific_neurons)
    num_layers = len(language_specific_neurons[0])

    print(f"Building complement masks: {num_languages} languages × {num_layers} layers")
    print(f"up_proj/gate_proj mask shape = ({intermediate_size}, {hidden_size})")
    print(f"down_proj mask shape = ({hidden_size}, {intermediate_size})")

    masks = {
        'up_proj': [],
        'gate_proj': [],
        'down_proj': []
    }

    for layer_idx in range(num_layers):
        # Collect union of all language-specific neurons for this layer
        union_set = set()

        for lang_idx in range(num_languages):
            neuron_ids = language_specific_neurons[lang_idx][layer_idx]
            if neuron_ids.numel() > 0:
                union_set.update(neuron_ids.tolist())

        union_list = sorted(list(union_set))
        
        # For up_proj and gate_proj: neurons are rows
        up_gate_mask = torch.ones(intermediate_size, hidden_size)
        if len(union_list) > 0:
            up_gate_mask[torch.tensor(union_list, dtype=torch.long), :] = 0
        
        # For down_proj: neurons are columns (transposed)
        down_mask = torch.ones(hidden_size, intermediate_size)
        if len(union_list) > 0:
            down_mask[:, torch.tensor(union_list, dtype=torch.long)] = 0

        masks['up_proj'].append(up_gate_mask)
        masks['gate_proj'].append(up_gate_mask.clone())  # Same mask
        masks['down_proj'].append(down_mask)

    torch.save(masks, save_path)
    print(f"\nSaved complement masks → {save_path}")

    return masks


model = AutoModelForCausalLM.from_pretrained("./llama3b")

hidden_size = model.model.layers[0].mlp.up_proj.weight.shape[1]  # 3072
intermediate_size_model = model.model.layers[0].mlp.up_proj.weight.shape[0]  # 8192

language_union_mask = build_language_union_mask(
    language_specific_neurons,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size_model,
    save_path=f"./language_masks/language_complement_mask_{TOP_RATE*100:.0f}_{FILTER_RATE*100:.0f}_{ACTIVATION_BAR_RATIO*100:.0f}_{int(WITH_EN)}.pt"
)


def build_lora_language_union_mask(language_specific_neurons,
                                   hidden_size,
                                   intermediate_size,
                                   rank,
                                   save_path="language_lora_mask.pt"):
    """
    Build complement mask for LoRA matrices (all neurons NOT in union of language neurons) per layer.
    
    In LoRA, the weight update is: ΔW = B @ A
        - B has shape (out_features, rank)
        - A has shape (rank, in_features)
    
    For row masking of ΔW (gate_proj, up_proj):
        - We mask rows of B, so B_mask has shape (intermediate_size, rank)
        - A is not masked, but we store a placeholder for consistency
    
    For column masking of ΔW (down_proj):
        - We mask columns of A, so A_mask has shape (rank, intermediate_size)
        - B is not masked, but we store a placeholder for consistency

    Mask values:
        1 → allow gradient (complement space - non-language neurons)
        0 → block gradient (language neurons)
    """

    num_languages = len(language_specific_neurons)
    num_layers = len(language_specific_neurons[0])

    print(f"\nBuilding LoRA complement masks: {num_languages} languages × {num_layers} layers, rank={rank}")
    print(f"gate_proj/up_proj B mask shape = ({intermediate_size}, {rank})")
    print(f"down_proj A mask shape = ({rank}, {intermediate_size})")

    masks = {
        # For gate_proj: mask rows of B (out_features=intermediate_size)
        'gate_proj.lora_B': [],
        'gate_proj.lora_A': [],  # No masking needed, all ones
        # For up_proj: mask rows of B (out_features=intermediate_size)
        'up_proj.lora_B': [],
        'up_proj.lora_A': [],  # No masking needed, all ones
        # For down_proj: mask columns of A (in_features=intermediate_size)
        'down_proj.lora_B': [],  # No masking needed, all ones
        'down_proj.lora_A': [],
    }

    for layer_idx in range(num_layers):
        # Collect union of all language-specific neurons for this layer
        union_set = set()

        for lang_idx in range(num_languages):
            neuron_ids = language_specific_neurons[lang_idx][layer_idx]
            if neuron_ids.numel() > 0:
                union_set.update(neuron_ids.tolist())

        union_list = sorted(list(union_set))
        complement_count = intermediate_size - len(union_list)

        if layer_idx < 3:  # Only print first few layers
            print(f"Layer {layer_idx} → {len(union_list)} language neurons (union), "
                  f"{complement_count} complement neurons")

        # === gate_proj and up_proj: mask rows of B ===
        # B shape: (intermediate_size, rank)
        # Masking rows of B → masks rows of ΔW = B @ A
        lora_B_row_mask = torch.ones(intermediate_size, rank)
        if len(union_list) > 0:
            lora_B_row_mask[torch.tensor(union_list, dtype=torch.long), :] = 0
        
        # A shape: (rank, hidden_size) - no masking needed for row masking of ΔW
        lora_A_no_mask = torch.ones(rank, hidden_size)
        
        masks['gate_proj.lora_B'].append(lora_B_row_mask)
        masks['gate_proj.lora_A'].append(lora_A_no_mask.clone())
        masks['up_proj.lora_B'].append(lora_B_row_mask.clone())
        masks['up_proj.lora_A'].append(lora_A_no_mask.clone())

        # === down_proj: mask columns of A ===
        # A shape: (rank, intermediate_size)
        # Masking columns of A → masks columns of ΔW = B @ A
        lora_A_col_mask = torch.ones(rank, intermediate_size)
        if len(union_list) > 0:
            lora_A_col_mask[:, torch.tensor(union_list, dtype=torch.long)] = 0
        
        # B shape: (hidden_size, rank) - no masking needed for column masking of ΔW
        lora_B_no_mask = torch.ones(hidden_size, rank)
        
        masks['down_proj.lora_B'].append(lora_B_no_mask)
        masks['down_proj.lora_A'].append(lora_A_col_mask)

    torch.save(masks, save_path)
    print(f"\nSaved LoRA complement masks → {save_path}")

    return masks

lora_rank = 64

lora_masks = build_lora_language_union_mask(
    language_specific_neurons,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size_model,
    rank=lora_rank,
    save_path=f"./language_masks/language_lora_complement_mask_{TOP_RATE*100:.0f}_{FILTER_RATE*100:.0f}_{ACTIVATION_BAR_RATIO*100:.0f}_{int(WITH_EN)}.pt"
)