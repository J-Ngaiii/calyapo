import torch

def compute_accuracy(logits, labels, ignore_index=-100):
    """
    Computes accuracy for the non-masked tokens (the completion).
    Works for both training and evaluation steps.

    logits: tensor of dimension (batch_size x sequence_length x llama_vocab_space)
        - for each indiviudal (first dim)
        - llm outputs a sequence of tokens as its predicted response (second dim)
        - each token is not yet collapsed down but a vector of logit scores over the whole vocab space (third dim)
    """
    # shift for Causal LM: token at N predicts label at N+1
    shift_logits = logits[..., :-1, :].contiguous() # '...' means keep all leading dimensions (eg. batch size and sequence length)
    shift_labels = labels[..., 1:].contiguous()
    
    # collapse down logit vectors to calculated predicted token at each point in the response sequence
    preds = torch.argmax(shift_logits, dim=-1)
    
    # create mask for tokens we actually want to predict (completion)
    mask = shift_labels != ignore_index
    
    if not mask.any():
        return 0.0, 0
        
    # perform elementwise comparison between LLM predictions and labels
    # handles for if LLM generates more than it's supposed to or generates token in the wrong spot
    correct = (preds[mask] == shift_labels[mask]).sum().item()
    total = mask.sum().item()
    
    return correct, total

def save_prediction_to_rollup(output_dir, local_rank, epoch, step, predictions, targets, tokenizer):
    """Streams decoded predictions to disk to avoid System RAM OOM."""
    rollup_path = os.path.join(output_dir, f"kl_rollup_rank_{local_rank}.jsonl")
    with open(rollup_path, "a") as f:
        # We only decode the first example in the batch to keep it fast, 
        # or you can loop through the batch if you need every single sample.
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
        
        for p, t in zip(decoded_preds, decoded_targets):
            entry = {"epoch": epoch, "step": step, "model": p.strip(), "human": t.strip()}
            f.write(json.dumps(entry) + "\n")

def compute_metrics(logits, labels, tokenizer, ignore_index=-100):
    """
    Modular helper for Calyapo-specific metrics.
    Compares model predictions only against unmasked tokens (the completion).
    """
    # Shift for Causal LM: logits at N predict label at N
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    preds = torch.argmax(shift_logits, dim=-1)
    
    # Mask where labels are NOT -100
    mask = shift_labels != ignore_index
    
    correct_count = (preds[mask] == shift_labels[mask]).sum().item()
    total_count = mask.sum().item()
    
    # Generate rollup strings for KL analysis
    rollups = []
    # We iterate batches to decode specific completions
    for i in range(labels.shape[0]):
        row_mask = shift_labels[i] != ignore_index
        if row_mask.any():
            p_str = tokenizer.decode(preds[i][row_mask], skip_special_tokens=True).strip()
            a_str = tokenizer.decode(shift_labels[i][row_mask], skip_special_tokens=True).strip()
            rollups.append({"model": p_str, "human": a_str})
            
    return {
        "correct_count": correct_count,
        "total_count": total_count,
        "rollups": rollups
    }