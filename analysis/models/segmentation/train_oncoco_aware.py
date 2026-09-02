#!/usr/bin/env python3
"""
OnCoCo-Aware Training mit Gumbel-Softmax + Two-Stage Loss.

Dieses Script trainiert das SaT-Modell direkt auf die OnCoCo-Metriken:
- Minimiere Repeat Rate (aufeinanderfolgende Spans mit gleichem Label)
- Maximiere OnCoCo Confidence (Classifier-Sicherheit)

Das Training ist SELF-SUPERVISED - es braucht keine Ground-Truth Satzgrenzen!
Stattdessen lernt das Modell, semantisch sinnvolle Segmentierungen zu finden.

Training-Ablauf:
================
1. SaT-Modell gibt Boundary-Wahrscheinlichkeiten für jede Token-Position
2. Gumbel-Softmax macht das Sampling differenzierbar
3. Soft-Span-Pooling extrahiert Span-Repräsentationen
4. OnCoCo-Classifier bewertet die Spans
5. Loss optimiert auf: wenig Repeats + hohe Confidence

Ergebnisse:
===========
- experiments/oncoco_aware/results/{experiment_name}.json  (Metriken)
- experiments/oncoco_aware/models/{experiment_name}/       (Adapter-Weights)
- experiments/oncoco_aware/checkpoints/                    (Checkpoints)
- experiments/oncoco_aware/comparison.json                 (Vergleich aller Modelle)

Usage:
    python scripts/train/train_oncoco_aware.py \
        --name "gumbel_lr1e-4_temp0.5" \
        --learning-rate 1e-4 \
        --temperature 0.5 \
        --epochs 5 \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
import signal
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import set_seed, get_linear_schedule_with_warmup
from tokenizers import AddedToken

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global flag für graceful shutdown
SHUTDOWN_REQUESTED = False

# OnCoCo Label Indices for Role Masking
# CO-Labels (Counselor): 0-39
# CL-Labels (Client): 40-67
COUNSELOR_LABEL_INDICES = list(range(0, 40))  # CO-*
CLIENT_LABEL_INDICES = list(range(40, 68))     # CL-*


def create_role_mask(role: str, num_labels: int = 68, device: str = 'cpu') -> torch.Tensor:
    """
    Create a mask for OnCoCo logits based on message role.

    As described in the OnCoCo paper:
    "We applied output masking by suppressing client-specific categories
    for counselor messages and vice versa at the level of the prediction logits."

    Args:
        role: 'counselor' or 'client'
        num_labels: Total number of OnCoCo labels (68)
        device: torch device

    Returns:
        Mask tensor [num_labels] with 0 for suppressed labels, 1 for allowed labels
    """
    mask = torch.zeros(num_labels, device=device)

    if role == 'counselor':
        # Counselor messages: only allow CO-* labels (0-39)
        for idx in COUNSELOR_LABEL_INDICES:
            if idx < num_labels:
                mask[idx] = 1.0
    elif role == 'client':
        # Client messages: only allow CL-* labels (40-67)
        for idx in CLIENT_LABEL_INDICES:
            if idx < num_labels:
                mask[idx] = 1.0
    else:
        # Unknown role: allow all labels (no masking)
        mask[:] = 1.0

    return mask


def apply_role_mask_to_logits(logits: torch.Tensor, role: str, device: str = 'cpu') -> torch.Tensor:
    """
    Apply role-based masking to OnCoCo logits.

    Sets logits of invalid labels to -inf so they have 0 probability after softmax.

    Args:
        logits: OnCoCo model output logits [batch, num_labels] or [num_labels]
        role: 'counselor' or 'client'
        device: torch device

    Returns:
        Masked logits with invalid labels set to -inf
    """
    num_labels = logits.shape[-1]
    mask = create_role_mask(role, num_labels, device)

    # Apply mask: set invalid labels to -inf
    # Where mask is 0, set logit to -inf
    masked_logits = logits.clone()
    masked_logits[..., mask == 0] = float('-inf')

    return masked_logits


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal from Slurm preemption."""
    global SHUTDOWN_REQUESTED
    logger.warning("=" * 60)
    logger.warning("SIGTERM received - Preemption detected!")
    logger.warning("Saving checkpoint and exiting gracefully...")
    logger.warning("=" * 60)
    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


# =============================================================================
#                          GUMBEL-SOFTMAX UTILITIES
# =============================================================================

def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Sigmoid für differenzierbares Binary Sampling.

    Args:
        logits: Input logits [batch, seq]
        temperature: Temperatur (niedriger = schärfer)
        hard: Wenn True, verwende Straight-Through Estimator

    Returns:
        Soft (oder hard) Boundary-Wahrscheinlichkeiten
    """
    # Gumbel noise
    U = torch.rand_like(logits)
    U = torch.clamp(U, 1e-10, 1 - 1e-10)  # Numerical stability
    gumbel_noise = -torch.log(-torch.log(U))

    # Gumbel-Sigmoid
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)

    if hard:
        # Straight-through estimator: forward = hard, backward = soft
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft

    return y_soft


def soft_span_pooling(
    hidden_states: torch.Tensor,
    boundary_probs: torch.Tensor,
    max_spans: int = 20,
    sigma: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Span Extraction mit Soft-Pooling.

    Statt harte Boundaries zu verwenden, gewichten wir die Hidden States
    basierend auf den Boundary-Wahrscheinlichkeiten.

    Args:
        hidden_states: Token embeddings [batch, seq, hidden]
        boundary_probs: Boundary probabilities [batch, seq]
        max_spans: Maximale Anzahl Spans

    Returns:
        span_embeddings: [batch, max_spans, hidden]
        span_mask: [batch, max_spans] - welche Spans sind valide
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    # Berechne kumulative Boundary-Wahrscheinlichkeiten
    # Dies gibt uns "Span-IDs" in soft Form
    cumsum_boundaries = torch.cumsum(boundary_probs, dim=1)  # [batch, seq]

    # Normalisiere auf [0, max_spans-1]
    max_cumsum = cumsum_boundaries[:, -1:].clamp(min=1.0)
    normalized_positions = (cumsum_boundaries / max_cumsum) * (max_spans - 1)

    # Soft assignment zu Span-Buckets
    span_indices = torch.arange(max_spans, device=device).float()  # [max_spans]

    # Berechne Soft-Assignment-Gewichte (Gaussian kernel)
    # [batch, seq, max_spans]
    distances = (normalized_positions.unsqueeze(-1) - span_indices.view(1, 1, -1)) ** 2
    # sigma controls sharpness: smaller = sharper span assignments
    gaussian_factor = 1.0 / (2.0 * sigma * sigma)
    weights = torch.exp(-distances * gaussian_factor)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    # Weighted pooling für jeden Span
    # [batch, max_spans, hidden]
    span_embeddings = torch.bmm(weights.transpose(1, 2), hidden_states)

    # Span mask basierend auf tatsächlicher Nutzung
    span_usage = weights.sum(dim=1)  # [batch, max_spans]
    span_mask = (span_usage > 0.1).float()

    return span_embeddings, span_mask


# =============================================================================
#                              DATASET
# =============================================================================

class OnCoCoAwareDataset(Dataset):
    """
    Dataset für OnCoCo-Aware Training.

    Im Gegensatz zum supervised Training brauchen wir hier KEINE Labels!
    Wir laden einfach die Texte und lassen das Modell selbst lernen,
    wo die besten Segmentierungen sind.

    Supports both old format (list of strings) and new format (list of dicts with 'text' and 'role').
    """

    def __init__(self, texts: List, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for item in tqdm(texts, desc="Tokenizing", leave=False):
            # Support both old (string) and new (dict) format
            if isinstance(item, dict):
                text = item.get('text', '').strip()
                role = item.get('role', 'unknown')
            else:
                text = item.strip()
                role = 'unknown'

            if len(text) < 20:  # Skip very short texts
                continue

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            self.examples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'text': text,
                'role': role
            })

        # Log role distribution
        role_counts = {}
        for ex in self.examples:
            r = ex['role']
            role_counts[r] = role_counts.get(r, 0) + 1
        logger.info(f"Created dataset with {len(self.examples)} examples")
        logger.info(f"Role distribution: {role_counts}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
#                           MODEL LOADING
# =============================================================================

def load_sat_model(model_name: str, lora_r: int = 8, lora_alpha: int = 16, full_finetune: bool = False):
    """Load SaT model with LoRA adapters or full fine-tuning.

    Args:
        model_name: HuggingFace model name
        lora_r: LoRA rank (ignored if full_finetune=True)
        lora_alpha: LoRA alpha (ignored if full_finetune=True)
        full_finetune: If True, train all parameters instead of using LoRA
    """
    from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification

    logger.info(f"Loading SaT model: {model_name}")
    logger.info(f"Training mode: {'Full Fine-Tuning' if full_finetune else f'LoRA (r={lora_r}, alpha={lora_alpha})'}")

    config = SubwordXLMConfig.from_pretrained(model_name)
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    model.config.base_model = "xlm-roberta-base"

    # Use xlm-roberta-base tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    model.resize_token_embeddings(len(tokenizer))

    if full_finetune:
        # Full fine-tuning: train all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Full fine-tuning mode: all parameters are trainable")
    else:
        # LoRA fine-tuning
        import adapters
        from adapters import LoRAConfig
        from adapters.models import MODEL_MIXIN_MAPPING
        from adapters.models.bert.mixin_bert import BertModelAdaptersMixin

        # Register adapter mixin
        MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin
        original_model_type = model.config.model_type
        model.config.model_type = "xlm-roberta"
        adapters.init(model)

        # LoRA config
        lora_config = LoRAConfig(r=lora_r, alpha=lora_alpha, dropout=0.1, leave_out=[])
        model.add_adapter("lora_adapter", config=lora_config, set_active=True)
        model.train_adapter("lora_adapter")
        model.config.model_type = original_model_type

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def load_oncoco_classifier(oncoco_path: str, device: str):
    """Load the OnCoCo classifier for reward computation."""
    logger.info(f"Loading OnCoCo classifier from {oncoco_path}")

    oncoco_clf = AutoModelForSequenceClassification.from_pretrained(oncoco_path)
    oncoco_tokenizer = AutoTokenizer.from_pretrained(oncoco_path)
    oncoco_clf.to(device)
    oncoco_clf.eval()

    # Freeze OnCoCo - wir trainieren nur das SaT-Modell!
    for param in oncoco_clf.parameters():
        param.requires_grad = False

    num_labels = oncoco_clf.config.num_labels
    logger.info(f"OnCoCo classifier loaded with {num_labels} labels")

    return oncoco_clf, oncoco_tokenizer


# =============================================================================
#                         ONCOCO-AWARE LOSS
# =============================================================================

class OnCoCoAwareLoss(nn.Module):
    """
    Simplified Loss für OnCoCo-Aware Training.

    Loss = repeat_rate - confidence

    Wobei:
    - repeat_rate: Anteil benachbarter Spans mit gleichem Label (zu minimieren)
    - confidence: Durchschnittliche OnCoCo-Konfidenz (zu maximieren)

    Diese vereinfachte Formel ist äquivalent zur gewichteten Version,
    da Experimente gezeigt haben, dass die Gewichtung keinen messbaren
    Unterschied macht.

    Zusätzlich: Boundary Regularization
        - Ermutigt klare Boundary-Entscheidungen (nahe 0 oder 1)
        - Verhindert "unsichere" Boundaries bei 0.5
    """

    def __init__(
        self,
        oncoco_clf: nn.Module,
        oncoco_tokenizer,
        diversity_weight: float = 1.0,  # Nicht mehr verwendet, für Backward-Kompatibilität
        confidence_weight: float = 1.0,  # Nicht mehr verwendet, für Backward-Kompatibilität
        boundary_reg_weight: float = 0.1,
        min_spans: int = 2,
        max_spans: int = 20
    ):
        super().__init__()
        self.oncoco_clf = oncoco_clf
        self.oncoco_tokenizer = oncoco_tokenizer
        self.diversity_weight = diversity_weight  # Deprecated
        self.confidence_weight = confidence_weight  # Deprecated
        self.boundary_reg_weight = boundary_reg_weight
        self.min_spans = min_spans
        self.max_spans = max_spans

    def forward(
        self,
        boundary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        temperature: float = 1.0,
        hard_gumbel: bool = False,
        pooling_sigma: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute OnCoCo-aware loss.

        Args:
            boundary_logits: SaT model output [batch, seq]
            hidden_states: Token embeddings [batch, seq, hidden]
            attention_mask: [batch, seq]
            input_ids: [batch, seq]
            tokenizer: SaT tokenizer for decoding spans
            temperature: Gumbel-Softmax temperature
            hard_gumbel: If True, use Straight-Through Estimator (hard boundaries)
            pooling_sigma: Gaussian sigma for soft span pooling (smaller = sharper)

        Returns:
            Dictionary with total_loss and individual components
        """
        batch_size, seq_len = boundary_logits.shape
        device = boundary_logits.device

        # 1. Gumbel-Softmax für differenzierbare Boundaries
        boundary_probs = gumbel_sigmoid(boundary_logits, temperature, hard=hard_gumbel)

        # Mask padding
        boundary_probs = boundary_probs * attention_mask

        # 2. Soft Span Pooling
        span_embeddings, span_mask = soft_span_pooling(
            hidden_states, boundary_probs, self.max_spans, sigma=pooling_sigma
        )

        # 3. Decode Spans und klassifiziere mit OnCoCo
        # Für jeden Batch: extrahiere Text-Spans und klassifiziere
        all_oncoco_probs = []
        all_confidences = []

        for b in range(batch_size):
            # Hard boundaries für Text-Extraktion (mit detach!)
            # SaT predicts "boundary AFTER this token" - same as wtpsplit
            hard_boundaries = (boundary_probs[b] > 0.5).float()
            boundary_indices = torch.where(hard_boundaries > 0.5)[0].tolist()
            seq_length = attention_mask[b].sum().int().item()

            # Convert to span end positions (wtpsplit style: idx + 1)
            boundary_positions = [0]  # Start of first span
            for idx in boundary_indices:
                end_pos = idx + 1  # Span includes the boundary token
                if end_pos < seq_length and end_pos not in boundary_positions:
                    boundary_positions.append(end_pos)
            if seq_length not in boundary_positions:
                boundary_positions.append(seq_length)
            boundary_positions = sorted(boundary_positions)

            # Extrahiere Span-Texte
            span_texts = []
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b])

            for i in range(min(len(boundary_positions) - 1, self.max_spans)):
                start, end = boundary_positions[i], boundary_positions[i + 1]
                if end <= start:
                    continue
                span_tokens = tokens[start:end]
                span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
                # Remove tokenizer special tokens (XLM-RoBERTa BOS/EOS)
                span_text = span_text.replace('<s>', '').replace('</s>', '').strip()
                if len(span_text) > 5:
                    span_texts.append(span_text)

            # Klassifiziere mit OnCoCo
            if len(span_texts) >= 2:
                oncoco_inputs = self.oncoco_tokenizer(
                    span_texts,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors='pt'
                ).to(device)

                with torch.no_grad():
                    oncoco_outputs = self.oncoco_clf(**oncoco_inputs)

                oncoco_probs = F.softmax(oncoco_outputs.logits, dim=-1)
                confidences = oncoco_probs.max(dim=-1).values

                all_oncoco_probs.append(oncoco_probs)
                all_confidences.append(confidences)

        # 4. Compute Losses

        # === DIVERSITY LOSS ===
        # Benachbarte Spans sollen unterschiedliche Labels haben
        diversity_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for oncoco_probs in all_oncoco_probs:
            if len(oncoco_probs) >= 2:
                # Cosine similarity zwischen benachbarten Span-Predictions
                similarities = F.cosine_similarity(
                    oncoco_probs[:-1], oncoco_probs[1:], dim=-1
                )
                diversity_loss = diversity_loss + similarities.mean()
                num_pairs += 1

        if num_pairs > 0:
            diversity_loss = diversity_loss / num_pairs

        # === CONFIDENCE LOSS ===
        # OnCoCo soll hohe Confidence haben
        confidence_loss = torch.tensor(0.0, device=device)

        if all_confidences:
            all_conf = torch.cat(all_confidences)
            confidence_loss = -all_conf.mean()  # Negative weil wir maximieren wollen

        # === BOUNDARY REGULARIZATION ===
        # Ermutigt klare Entscheidungen (nahe 0 oder 1)
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        p = boundary_probs * attention_mask
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        boundary_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        boundary_reg = boundary_entropy.sum() / attention_mask.sum()

        # === SPAN COUNT REGULARIZATION ===
        # Ermutigt eine vernünftige Anzahl an Spans
        avg_boundaries = boundary_probs.sum(dim=1).mean()
        span_count_target = 5.0  # Ziel: ~5 Spans pro Text
        span_count_loss = (avg_boundaries - span_count_target) ** 2 * 0.01

        # === TOTAL LOSS ===
        # Vereinfachte Loss-Formel: loss = diversity_loss + confidence_loss
        # Equivalent zu: loss = repeat_rate - confidence
        # Die Gewichtung (diversity_weight, confidence_weight) wird nicht mehr verwendet,
        # da Experimente gezeigt haben, dass sie keinen messbaren Unterschied macht.
        total_loss = (
            diversity_loss +
            confidence_loss +
            self.boundary_reg_weight * boundary_reg +
            span_count_loss
        )

        return {
            'total_loss': total_loss,
            'diversity_loss': diversity_loss,
            'confidence_loss': confidence_loss,
            'boundary_reg': boundary_reg,
            'span_count_loss': span_count_loss,
            'avg_boundaries': avg_boundaries,
            'avg_confidence': -confidence_loss if all_confidences else torch.tensor(0.0)
        }


# =============================================================================
#                            TRAINING LOOP
# =============================================================================

def train_epoch(
    model,
    oncoco_loss_fn: OnCoCoAwareLoss,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    tokenizer,
    temperature: float,
    epoch: int,
    hard_gumbel: bool = False,
    pooling_sigma: float = 0.5
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()

    total_loss = 0
    total_diversity = 0
    total_confidence = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if SHUTDOWN_REQUESTED:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Forward durch SaT-Modell mit hidden states
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        boundary_logits = outputs.logits[:, :, 0]  # [batch, seq]

        # Hidden states für Span-Pooling
        # Wir nutzen die letzten Hidden States des Encoders
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            # Fallback: Verwende Output-Logits als Proxy
            hidden_states = outputs.logits

        # OnCoCo-Aware Loss
        loss_dict = oncoco_loss_fn(
            boundary_logits=boundary_logits,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
            tokenizer=tokenizer,
            temperature=temperature,
            hard_gumbel=hard_gumbel,
            pooling_sigma=pooling_sigma
        )

        loss = loss_dict['total_loss']

        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_diversity += loss_dict['diversity_loss'].item()
        total_confidence += loss_dict['avg_confidence'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'div': f'{loss_dict["diversity_loss"].item():.3f}',
            'conf': f'{loss_dict["avg_confidence"].item():.3f}'
        })

    return {
        'loss': total_loss / max(num_batches, 1),
        'diversity_loss': total_diversity / max(num_batches, 1),
        'avg_confidence': total_confidence / max(num_batches, 1)
    }


def evaluate_model(
    model,
    tokenizer,
    test_texts: List,
    oncoco_clf,
    oncoco_tokenizer,
    device: str,
    num_samples: int = 100,
    save_spans: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on test set.

    Berechnet die echten Metriken:
    - Repeat Rate
    - Average Confidence
    - Combined Score

    Supports both old format (list of strings) and new format (list of dicts with 'text' and 'role').
    When role is available, applies role-based masking to OnCoCo logits.

    If save_spans=True, returns detailed span information for each message to allow
    metric recalculation without re-running the model.
    """
    model.eval()

    # Per-message metrics (repeat rate is calculated WITHIN each message, not across messages)
    all_confidences = []
    total_spans = 0
    num_repeats = 0
    total_pairs = 0

    # Store detailed results for each message (for later re-evaluation)
    messages_data = []

    with torch.no_grad():
        for msg_idx, item in enumerate(tqdm(test_texts[:num_samples], desc="Evaluating")):
            # Support both old (string) and new (dict) format
            if isinstance(item, dict):
                text = item.get('text', '')
                role = item.get('role', 'unknown')
            else:
                text = item
                role = 'unknown'

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            outputs = model(**encoding)
            probs = torch.sigmoid(outputs.logits[0, :, 0]).cpu().numpy()

            # Find boundaries
            # SaT predicts "boundary AFTER this token" - probs[i] > 0.5 means split AFTER token i
            # wtpsplit's indices_to_sentences does: idx = idx + 1, then text[offset:idx]
            # So a boundary at position 5 means the span ends at position 6 (inclusive of token 5)
            seq_len = int(encoding['attention_mask'].sum().item())
            boundary_indices = np.where(probs[:seq_len] > 0.5)[0].tolist()

            # Convert to span end positions (wtpsplit style: idx + 1)
            boundaries = [0]  # Start of first span
            for idx in boundary_indices:
                end_pos = idx + 1  # Span includes the boundary token
                if end_pos < seq_len and end_pos not in boundaries:
                    boundaries.append(end_pos)
            if seq_len not in boundaries:
                boundaries.append(seq_len)
            boundaries = sorted(boundaries)

            # Extract spans and labels FOR THIS MESSAGE ONLY
            tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            message_labels = []  # Labels only for this message
            message_spans = []   # Detailed span info for this message

            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if end <= start:
                    continue

                span_tokens = tokens[start:end]
                span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
                # Remove tokenizer special tokens (XLM-RoBERTa BOS/EOS)
                span_text = span_text.replace('<s>', '').replace('</s>', '').strip()

                if len(span_text) < 5:
                    continue

                # Classify with OnCoCo
                inputs = oncoco_tokenizer(
                    span_text,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(device)

                clf_outputs = oncoco_clf(**inputs)

                # Apply role-based masking to logits
                # This suppresses invalid labels (CO-* for clients, CL-* for counselors)
                masked_logits = apply_role_mask_to_logits(clf_outputs.logits, role, device)

                probs_oncoco = torch.softmax(masked_logits, dim=-1)
                confidence = probs_oncoco.max().item()
                label_id = probs_oncoco.argmax().item()

                if hasattr(oncoco_clf.config, 'id2label'):
                    label = oncoco_clf.config.id2label[label_id]
                else:
                    label = f"label_{label_id}"

                message_labels.append(label)
                all_confidences.append(confidence)
                total_spans += 1

                # Store span details
                if save_spans:
                    message_spans.append({
                        'text': span_text,
                        'label': label,
                        'confidence': float(confidence)
                    })

            # Calculate repeats WITHIN this message only
            # (consecutive spans with same label within the same message)
            message_repeats = 0
            message_pairs = len(message_labels) - 1 if len(message_labels) > 1 else 0
            for i in range(len(message_labels) - 1):
                total_pairs += 1
                if message_labels[i] == message_labels[i + 1]:
                    num_repeats += 1
                    message_repeats += 1

            # Store message data
            if save_spans:
                messages_data.append({
                    'message_id': msg_idx,
                    'role': role,
                    'spans': message_spans,
                    'num_spans': len(message_spans),
                    'num_pairs': message_pairs,
                    'num_repeats': message_repeats
                })

    # Calculate final metrics
    repeat_rate = num_repeats / max(total_pairs, 1)
    avg_confidence = np.mean(all_confidences) if all_confidences else 0

    # Harmonic Score: H = 2 * (1-RR) * Conf / ((1-RR) + Conf)
    # Wertebereich: 0 bis 1 (höher = besser)
    # Bestraft Ungleichgewicht zwischen den beiden Metriken
    diversity_rate = 1.0 - repeat_rate  # Konvertiere repeat_rate zu diversity_rate
    if diversity_rate + avg_confidence > 0:
        harmonic_score = 2 * diversity_rate * avg_confidence / (diversity_rate + avg_confidence)
    else:
        harmonic_score = 0.0

    # Calculate span length statistics
    all_span_lengths = []
    for msg in messages_data:
        for span in msg.get('spans', []):
            all_span_lengths.append(len(span.get('text', '')))

    avg_span_length = np.mean(all_span_lengths) if all_span_lengths else 0
    std_span_length = np.std(all_span_lengths) if len(all_span_lengths) > 1 else 0

    result = {
        'repeat_rate': float(repeat_rate),
        'avg_confidence': float(avg_confidence),
        'harmonic_score': float(harmonic_score),
        'total_spans': total_spans,
        'num_repeats': num_repeats,
        'total_pairs': total_pairs,
        'total_messages': num_samples,
        'spans_per_message': total_spans / num_samples if num_samples > 0 else 0,
        'avg_span_length': float(avg_span_length),
        'std_span_length': float(std_span_length)
    }

    # Add detailed message data if requested
    if save_spans:
        result['messages'] = messages_data

    return result


def format_unified_result(
    name: str,
    eval_results: Dict[str, Any],
    args,
    training_time: float,
    train_history: List[Dict],
    eval_epoch: float
) -> Dict[str, Any]:
    """
    Format evaluation results in the unified format.

    This format is consistent across SaT training results and LLM baselines,
    making it easier to compare and analyze results.
    """
    from datetime import datetime

    # Convert flat messages to conversation structure
    # Since we evaluate on Säule 5 (which has conversation structure),
    # we group all messages into a single "conversation" for now
    # In the future, this could be enhanced to preserve actual conversation IDs
    messages_data = eval_results.get('messages', [])

    # Add message-level metrics to each message
    conversation_messages = []
    for msg in messages_data:
        spans = msg.get('spans', [])
        num_spans = len(spans)
        num_pairs = max(0, num_spans - 1)

        # Calculate repeat rate within message
        num_repeats = 0
        for i in range(1, num_spans):
            if spans[i].get('label') == spans[i-1].get('label'):
                num_repeats += 1

        # Calculate avg confidence for message
        confidences = [s.get('confidence', 0) for s in spans]
        msg_avg_conf = sum(confidences) / len(confidences) if confidences else 0

        conversation_messages.append({
            'message_id': msg.get('message_id', 0),
            'role': msg.get('role', 'unknown'),
            'metrics': {
                'num_spans': num_spans,
                'num_pairs': num_pairs,
                'num_repeats': num_repeats,
                'repeat_rate': round(num_repeats / num_pairs, 6) if num_pairs > 0 else 0.0,
                'avg_confidence': round(msg_avg_conf, 6)
            },
            'spans': spans
        })

    # Build conversation with aggregated metrics
    conv_total_spans = eval_results.get('total_spans', 0)
    conv_total_pairs = eval_results.get('total_pairs', 0)
    conv_total_repeats = eval_results.get('num_repeats', 0)
    conv_repeat_rate = eval_results.get('repeat_rate', 0)
    conv_avg_conf = eval_results.get('avg_confidence', 0)
    conv_harmonic = eval_results.get('harmonic_score', 0)

    conversations = [{
        'conversation_id': 0,
        'source_file': 'saeule_5_test_set',
        'metrics': {
            'total_spans': conv_total_spans,
            'total_pairs': conv_total_pairs,
            'total_repeats': conv_total_repeats,
            'repeat_rate': round(conv_repeat_rate, 6),
            'avg_confidence': round(conv_avg_conf, 6),
            'harmonic_score': round(conv_harmonic, 6)
        },
        'messages': conversation_messages
    }]

    # Build unified result
    result = {
        'name': name,
        'method': 'sat_finetuned',
        'created_at': datetime.now().isoformat(),

        'model': {
            'name': args.model_name,
            'provider': 'huggingface',
            'type': 'sat'
        },

        'training': {
            'learning_rate': args.learning_rate,
            'temperature': args.temperature,
            'full_finetune': args.full_finetune,
            'lora_r': args.lora_r if not args.full_finetune else None,
            'lora_alpha': args.lora_alpha if not args.full_finetune else None,
            'epochs_trained': args.epochs,
            'eval_epoch': eval_epoch,
            'batch_size': args.batch_size,
            'diversity_weight': args.diversity_weight,
            'confidence_weight': args.confidence_weight,
            'training_time_seconds': round(training_time, 2),
            'train_history': train_history
        },

        'metrics': {
            'repeat_rate': round(eval_results.get('repeat_rate', 0), 6),
            'avg_confidence': round(eval_results.get('avg_confidence', 0), 6),
            'harmonic_score': round(eval_results.get('harmonic_score', 0), 6),
            'total_spans': eval_results.get('total_spans', 0),
            'total_pairs': eval_results.get('total_pairs', 0),
            'total_repeats': eval_results.get('num_repeats', 0),
            'total_messages': eval_results.get('total_messages', 0),
            'total_conversations': 1,  # Currently we treat all as one conversation
            'spans_per_message': round(eval_results.get('spans_per_message', 0), 2),
            'avg_span_length': round(eval_results.get('avg_span_length', 0), 2),
            'std_span_length': round(eval_results.get('std_span_length', 0), 2)
        },

        'conversations': conversations
    }

    return result


# =============================================================================
#                               MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OnCoCo-Aware Training")
    parser.add_argument("--name", type=str, help="Experiment name (deprecated, use --output-prefix)")
    parser.add_argument("--output-prefix", type=str, help="Output file prefix (overrides --name)")
    parser.add_argument("--model", type=str, help="Model short name (e.g., sat-9l)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Gumbel temperature")
    parser.add_argument("--hard-gumbel", action="store_true", help="Use Straight-Through Estimator (hard boundaries)")
    parser.add_argument("--pooling-sigma", type=float, default=0.5, help="Gaussian sigma for soft span pooling (smaller = sharper)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (default: 2*r)")
    parser.add_argument("--full-finetune", action="store_true", help="Full fine-tuning instead of LoRA")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--eval-epochs", type=str, default="1,3,5", help="Epochs to evaluate at (comma-separated or space-separated)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--diversity-weight", type=float, default=0.7, help="Diversity loss weight")
    parser.add_argument("--confidence-weight", type=float, default=0.3, help="Confidence loss weight")
    parser.add_argument("--model-name", type=str, default=None, help="Full model name (e.g., segment-any-text/sat-3l)")
    parser.add_argument("--data-path", type=str, default="data/sat_oncoco_train/oncoco_data_full.pth")
    parser.add_argument("--oncoco-path", type=str, default="model/oncoco")
    parser.add_argument("--output-dir", type=str, default="experiments/oncoco_aware")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-model", action="store_true", help="Don't save model checkpoints (only JSON results)")

    args = parser.parse_args()

    # Vollständige Reproduzierbarkeit auf GPU
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Optional: torch.use_deterministic_algorithms(True) - kann bei manchen Ops Fehler werfen

    # Handle model name: --model (short name) or --model-name (full path)
    MODEL_MAP = {
        "sat-1l": "segment-any-text/sat-1l",
        "sat-3l": "segment-any-text/sat-3l",
        "sat-6l": "segment-any-text/sat-6l",
        "sat-9l": "segment-any-text/sat-9l",
        "sat-12l": "segment-any-text/sat-12l",
        "sat-1l-sm": "segment-any-text/sat-1l-sm",
        "sat-3l-sm": "segment-any-text/sat-3l-sm",
        "sat-6l-sm": "segment-any-text/sat-6l-sm",
        "sat-9l-sm": "segment-any-text/sat-9l-sm",
        "sat-12l-sm": "segment-any-text/sat-12l-sm",
    }

    if args.model:
        args.model_name = MODEL_MAP.get(args.model, f"segment-any-text/{args.model}")
    elif not args.model_name:
        args.model_name = "segment-any-text/sat-3l"  # Default

    # Handle experiment name
    experiment_name = args.output_prefix or args.name
    if not experiment_name:
        # Generate from parameters
        model_short = args.model or args.model_name.split("/")[-1]
        experiment_name = f"{model_short}_lr{args.learning_rate}_temp{args.temperature}_r{args.lora_r}"

    # Set default LoRA alpha
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    logger.info("=" * 60)
    logger.info("OnCoCo-Aware Training mit Gumbel-Softmax")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Hard Gumbel: {args.hard_gumbel}")
    logger.info(f"Pooling Sigma: {args.pooling_sigma}")
    logger.info(f"Full Fine-Tune: {args.full_finetune}")
    if not args.full_finetune:
        logger.info(f"LoRA r/alpha: {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Eval at epochs: {args.eval_epochs}")
    logger.info(f"Diversity Weight: {args.diversity_weight}")
    logger.info(f"Confidence Weight: {args.confidence_weight}")
    logger.info("=" * 60)

    # Parse eval epochs (support both comma-separated and space-separated, and floats like 1.5)
    eval_epochs_str = args.eval_epochs.replace(',', ' ')
    eval_epochs = [float(e.strip()) for e in eval_epochs_str.split() if e.strip()]
    logger.info(f"Will evaluate and save at epochs: {eval_epochs}")

    # Update args.name for downstream use
    args.name = experiment_name

    # Paths
    output_dir = Path(args.output_dir)
    result_path = output_dir / "results" / f"{args.name}.json"
    model_path = output_dir / "models" / args.name / "adapter"
    checkpoint_path = output_dir / "checkpoints" / f"{args.name}_checkpoint.pt"

    # Create directories
    result_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    if result_path.exists():
        logger.info(f"Experiment already completed: {result_path}")
        return

    # Load data
    logger.info("Loading data...")
    data = torch.load(args.data_path)
    oncoco_data = data["de"]["sentence"]["oncoco"]
    train_texts = oncoco_data["meta"]["train_data"]
    test_texts = oncoco_data["data"]

    logger.info(f"Train texts: {len(train_texts)}")
    logger.info(f"Test texts: {len(test_texts)}")

    # Load models
    model, tokenizer = load_sat_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        full_finetune=args.full_finetune
    )
    model.to(args.device)

    oncoco_clf, oncoco_tokenizer = load_oncoco_classifier(args.oncoco_path, args.device)

    # Create dataset with deterministic shuffling
    train_dataset = OnCoCoAwareDataset(train_texts, tokenizer)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)

    # Create loss function
    oncoco_loss_fn = OnCoCoAwareLoss(
        oncoco_clf=oncoco_clf,
        oncoco_tokenizer=oncoco_tokenizer,
        diversity_weight=args.diversity_weight,
        confidence_weight=args.confidence_weight
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop with multi-epoch evaluation (supports half epochs like 1.5, 2.5)
    logger.info("Starting training...")
    start_time = time.time()
    train_history = []
    all_epoch_results = {}

    # Sort eval epochs for proper checkpoint handling
    eval_epochs_sorted = sorted(eval_epochs)

    # Helper function to evaluate and save
    def evaluate_and_save(current_progress: float):
        """Evaluate at current progress point (e.g., 1.0, 1.5, 2.0)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING AT EPOCH {current_progress}")
        logger.info(f"{'='*60}")

        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_texts=test_texts,
            oncoco_clf=oncoco_clf,
            oncoco_tokenizer=oncoco_tokenizer,
            device=args.device,
            num_samples=len(test_texts),  # Evaluate all test texts
            save_spans=True  # Save spans for later re-evaluation
        )

        # Store results for this epoch
        all_epoch_results[current_progress] = eval_results

        # Create epoch-specific result file name
        base_name = args.name
        epoch_result_path = output_dir / f"{base_name}_ep{current_progress}.json"
        epoch_model_path = output_dir / "models" / f"{base_name}_ep{current_progress}" / "adapter"
        epoch_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Format results in unified format
        unified_results = format_unified_result(
            name=f"{base_name}_ep{current_progress}",
            eval_results=eval_results,
            args=args,
            training_time=time.time() - start_time,
            train_history=train_history.copy(),
            eval_epoch=current_progress
        )

        with open(epoch_result_path, 'w') as f:
            json.dump(unified_results, f, indent=2, ensure_ascii=False)

        # Save model checkpoint for this epoch (unless --no-save-model)
        if not args.no_save_model:
            if args.full_finetune:
                # Save full model for full fine-tuning
                model.save_pretrained(str(epoch_model_path.parent))
                tokenizer.save_pretrained(str(epoch_model_path.parent))
            else:
                # Save only adapter for LoRA
                model.save_adapter(str(epoch_model_path), "lora_adapter")

        metrics = unified_results['metrics']
        logger.info(f"Epoch {current_progress} Results:")
        logger.info(f"  Repeat Rate:    {metrics['repeat_rate']*100:.2f}%")
        logger.info(f"  Avg Confidence: {metrics['avg_confidence']*100:.2f}%")
        logger.info(f"  Harmonic Score: {metrics['harmonic_score']:.4f}")
        logger.info(f"  Spans/Message:  {metrics['spans_per_message']:.2f}")
        logger.info(f"  Avg Span Len:   {metrics['avg_span_length']:.0f}±{metrics['std_span_length']:.0f}")
        logger.info(f"  Saved to: {epoch_result_path}")
        logger.info(f"  Adapter: {epoch_model_path}")
        logger.info(f"{'='*60}\n")

    for epoch in range(args.epochs):
        current_epoch = epoch + 1

        if SHUTDOWN_REQUESTED:
            logger.warning("Shutdown requested, saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history
            }, checkpoint_path)
            break

        model.train()
        total_loss = 0
        total_diversity = 0
        total_confidence = 0
        num_batches = 0
        total_batches = len(train_loader)
        half_epoch_batch = total_batches // 2  # Batch at which we're at 0.5 epoch

        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            if SHUTDOWN_REQUESTED:
                break

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)

            optimizer.zero_grad()

            # Forward durch SaT-Modell mit hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            boundary_logits = outputs.logits[:, :, 0]  # [batch, seq]

            # Hidden states für Span-Pooling
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.logits

            # OnCoCo-Aware Loss
            loss_dict = oncoco_loss_fn(
                boundary_logits=boundary_logits,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                input_ids=input_ids,
                tokenizer=tokenizer,
                temperature=args.temperature,
                hard_gumbel=args.hard_gumbel,
                pooling_sigma=args.pooling_sigma
            )

            loss = loss_dict['total_loss']

            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_diversity += loss_dict['diversity_loss'].item()
            total_confidence += loss_dict['avg_confidence'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'div': f'{loss_dict["diversity_loss"].item():.3f}',
                'conf': f'{loss_dict["avg_confidence"].item():.3f}'
            })

            # Check for half-epoch evaluation (e.g., 1.5, 2.5, 3.5, 4.5)
            if batch_idx == half_epoch_batch:
                half_epoch_point = current_epoch - 0.5
                if half_epoch_point in eval_epochs_sorted:
                    evaluate_and_save(half_epoch_point)
                    model.train()  # Back to training mode

        # End of epoch metrics
        epoch_metrics = {
            'loss': total_loss / max(num_batches, 1),
            'diversity_loss': total_diversity / max(num_batches, 1),
            'avg_confidence': total_confidence / max(num_batches, 1)
        }
        train_history.append(epoch_metrics)
        logger.info(f"Epoch {current_epoch}: loss={epoch_metrics['loss']:.4f}, "
                   f"diversity={epoch_metrics['diversity_loss']:.4f}, "
                   f"confidence={epoch_metrics['avg_confidence']:.4f}")

        # Check if we should evaluate at this full epoch
        if float(current_epoch) in eval_epochs_sorted:
            evaluate_and_save(float(current_epoch))

    training_time = time.time() - start_time

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - SUMMARY OF ALL EPOCHS")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Config: lr={args.learning_rate}, temp={args.temperature}, r={args.lora_r}")
    logger.info(f"Total Training Time: {training_time/60:.1f} min")
    logger.info("")
    logger.info(f"{'Epoch':<8} {'Repeat%':<10} {'Conf%':<10} {'H-Score':<10}")
    logger.info("-" * 40)
    for ep, res in sorted(all_epoch_results.items()):
        logger.info(f"ep{ep:<6} {res['repeat_rate']*100:<10.2f} {res['avg_confidence']*100:<10.2f} {res['harmonic_score']:<10.4f}")
    logger.info("=" * 60)

    # Find best epoch
    if all_epoch_results:
        best_epoch = max(all_epoch_results.keys(), key=lambda e: all_epoch_results[e]['harmonic_score'])
        best_res = all_epoch_results[best_epoch]
        logger.info(f"BEST: Epoch {best_epoch} with Harmonic Score {best_res['harmonic_score']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
