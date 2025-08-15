"""
TOAI-MR: Translator-Oriented AI for Medical Record Reconciliation

This module implements the core TOAI-MR model architecture with specialized
components for medical record translation between EHR systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from transformers import AutoTokenizer, AutoModel
import json
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class MedicalTokenizer:
    """Specialized tokenizer for medical domain with EHR-specific vocabulary."""
    
    def __init__(self, vocab_size: int = 50000):
        """Initialize medical tokenizer."""
        self.vocab_size = vocab_size
        
        # Medical domain vocabulary
        self.medical_vocab = self._build_medical_vocab()
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Add medical tokens to vocabulary
        self.base_tokenizer.add_tokens(list(self.medical_vocab.keys()))
        
    def _build_medical_vocab(self) -> Dict[str, int]:
        """Build medical domain vocabulary."""
        medical_terms = [
            # FHIR resource types
            "Patient", "Encounter", "Condition", "MedicationRequest", "Observation",
            "Procedure", "DiagnosticReport", "AllergyIntolerance", "CarePlan",
            
            # HL7 segments
            "MSH", "PID", "PV1", "OBX", "ORC", "RXA", "DG1", "PR1",
            
            # Medical abbreviations
            "BP", "HR", "RR", "O2SAT", "TEMP", "BMI", "A1C", "CBC", "CMP",
            "ECG", "EKG", "CT", "MRI", "XRAY", "US", "PO", "IV", "IM", "SC",
            
            # Common medical terms
            "hypertension", "diabetes", "hyperlipidemia", "asthma", "COPD",
            "depression", "anxiety", "arthritis", "osteoporosis", "CAD",
            "CHF", "AFib", "DVT", "PE", "MI", "CVA", "TIA", "UTI", "pneumonia",
            
            # Medication terms
            "mg", "mcg", "ml", "tablet", "capsule", "injection", "daily",
            "twice", "three", "times", "PRN", "QID", "BID", "TID", "QD",
            
            # Units and measurements
            "mmHg", "bpm", "kg", "lbs", "cm", "ft", "in", "celsius", "fahrenheit"
        ]
        
        return {term: i for i, term in enumerate(medical_terms)}
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode text with medical tokenization."""
        return self.base_tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=True)

class MedicalKnowledgeEmbedding(nn.Module):
    """Embedding layer that incorporates medical domain knowledge."""
    
    def __init__(self, vocab_size: int, embed_dim: int, medical_vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Standard word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Medical concept embeddings
        self.medical_embeddings = nn.Embedding(medical_vocab_size, embed_dim)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(512, embed_dim)
        
        # Medical structure embeddings (for EHR structure awareness)
        self.structure_embeddings = nn.Embedding(100, embed_dim)  # 100 different structure types
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                structure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with medical knowledge integration."""
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = word_embeds + position_embeds
        
        # Add structure embeddings if provided
        if structure_ids is not None:
            structure_embeds = self.structure_embeddings(structure_ids)
            embeddings = embeddings + structure_embeds
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MedicalAttention(nn.Module):
    """Multi-head attention with medical context awareness."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Medical context attention weights
        self.medical_context_weight = nn.Parameter(torch.randn(1, num_heads, 1, 1))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                medical_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with medical context attention."""
        batch_size, seq_len, embed_dim = query.size()
        
        # Transform inputs
        Q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply medical context weighting
        if medical_mask is not None:
            medical_boost = self.medical_context_weight * medical_mask.unsqueeze(1).unsqueeze(1)
            scores = scores + medical_boost
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        output = self.out_proj(attn_output)
        return output

class MedicalTransformerLayer(nn.Module):
    """Transformer layer with medical domain adaptations."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.self_attn = MedicalAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Medical knowledge injection layer
        self.medical_knowledge_gate = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, medical_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with medical knowledge integration."""
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, medical_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Medical knowledge injection
        medical_gate = torch.sigmoid(self.medical_knowledge_gate(x))
        x = x * medical_gate
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TOAIMREncoder(nn.Module):
    """TOAI-MR Encoder with medical domain specialization."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Medical embeddings
        self.embeddings = MedicalKnowledgeEmbedding(vocab_size, embed_dim, 1000)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MedicalTransformerLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Medical context analyzer
        self.medical_context_analyzer = nn.Linear(embed_dim, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode input with medical context awareness."""
        # Get embeddings
        x = self.embeddings(input_ids)
        
        # Generate medical context mask
        medical_mask = None
        if attention_mask is not None:
            medical_context_scores = self.medical_context_analyzer(x)
            medical_mask = torch.sigmoid(medical_context_scores).squeeze(-1) * attention_mask
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, medical_mask)
        
        return {
            "last_hidden_state": x,
            "medical_context_mask": medical_mask
        }

class TOAIMRDecoder(nn.Module):
    """TOAI-MR Decoder with EHR format generation capabilities."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Medical embeddings
        self.embeddings = MedicalKnowledgeEmbedding(vocab_size, embed_dim, 1000)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            MedicalTransformerLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Cross-attention layers for encoder-decoder attention
        self.cross_attention_layers = nn.ModuleList([
            MedicalAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # EHR format constraint layer
        self.format_constraint_layer = nn.Linear(embed_dim, 10)  # 10 different EHR formats
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(embed_dim, 1)
        
    def forward(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Decode with EHR format awareness."""
        # Get embeddings
        x = self.embeddings(input_ids)
        
        # Pass through decoder layers with cross-attention
        for i, (layer, cross_attn) in enumerate(zip(self.layers, self.cross_attention_layers)):
            # Self-attention
            x = layer(x, decoder_attention_mask)
            
            # Cross-attention to encoder outputs
            cross_attn_output = cross_attn(x, encoder_outputs, encoder_outputs, encoder_attention_mask)
            x = x + cross_attn_output
        
        # Generate outputs
        logits = self.output_projection(x)
        format_logits = self.format_constraint_layer(x)
        confidence_scores = torch.sigmoid(self.confidence_head(x))
        
        return {
            "logits": logits,
            "format_logits": format_logits,
            "confidence_scores": confidence_scores
        }

class TOAIMRModel(nn.Module):
    """Complete TOAI-MR model for medical record translation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        vocab_size = config.get("vocab_size", 50000)
        embed_dim = config.get("embed_dim", 1024)
        num_encoder_layers = config.get("num_encoder_layers", 24)
        num_decoder_layers = config.get("num_decoder_layers", 24)
        num_heads = config.get("num_heads", 16)
        ff_dim = config.get("ff_dim", 4096)
        
        # Initialize tokenizer
        self.tokenizer = MedicalTokenizer(vocab_size)
        
        # Initialize encoder and decoder
        self.encoder = TOAIMREncoder(vocab_size, embed_dim, num_encoder_layers, num_heads, ff_dim)
        self.decoder = TOAIMRDecoder(vocab_size, embed_dim, num_decoder_layers, num_heads, ff_dim)
        
        # Medical validation layer
        self.medical_validator = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, source_input_ids: torch.Tensor, target_input_ids: torch.Tensor,
                source_attention_mask: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Encode source
        encoder_outputs = self.encoder(source_input_ids, source_attention_mask)
        
        # Decode target
        decoder_outputs = self.decoder(
            target_input_ids,
            encoder_outputs["last_hidden_state"],
            encoder_outputs.get("medical_context_mask"),
            target_attention_mask
        )
        
        # Medical validation
        validation_scores = self.medical_validator(encoder_outputs["last_hidden_state"])
        
        return {
            "logits": decoder_outputs["logits"],
            "format_logits": decoder_outputs["format_logits"],
            "confidence_scores": decoder_outputs["confidence_scores"],
            "validation_scores": validation_scores,
            "encoder_outputs": encoder_outputs
        }
    
    def generate(self, source_input_ids: torch.Tensor, source_attention_mask: Optional[torch.Tensor] = None,
                 max_length: int = 512, target_format: str = "fhir") -> Dict[str, Any]:
        """Generate translation for inference."""
        self.eval()
        
        with torch.no_grad():
            # Encode source
            encoder_outputs = self.encoder(source_input_ids, source_attention_mask)
            
            # Initialize generation
            batch_size = source_input_ids.size(0)
            device = source_input_ids.device
            
            # Start with BOS token
            generated_ids = torch.full((batch_size, 1), self.tokenizer.base_tokenizer.cls_token_id, 
                                     dtype=torch.long, device=device)
            
            # Generate tokens iteratively
            for _ in range(max_length - 1):
                decoder_outputs = self.decoder(
                    generated_ids,
                    encoder_outputs["last_hidden_state"],
                    encoder_outputs.get("medical_context_mask")
                )
                
                # Get next token probabilities
                next_token_logits = decoder_outputs["logits"][:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token_ids = torch.multinomial(next_token_probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)
                
                # Check for EOS token
                if (next_token_ids == self.tokenizer.base_tokenizer.sep_token_id).all():
                    break
            
            # Decode generated text
            generated_text = [
                self.tokenizer.decode(ids) for ids in generated_ids
            ]
            
            # Get confidence scores
            final_decoder_outputs = self.decoder(
                generated_ids,
                encoder_outputs["last_hidden_state"],
                encoder_outputs.get("medical_context_mask")
            )
            
            confidence_scores = final_decoder_outputs["confidence_scores"].mean(dim=1)
            
            return {
                "generated_text": generated_text,
                "generated_ids": generated_ids,
                "confidence_scores": confidence_scores,
                "target_format": target_format
            }
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and configuration."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str) -> "TOAIMRModel":
        """Load model from directory."""
        import os
        
        # Load configuration
        with open(os.path.join(model_directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(config)
        
        # Load state dict
        state_dict = torch.load(os.path.join(model_directory, "pytorch_model.bin"), 
                               map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model

class TOAIMRLoss(nn.Module):
    """Custom loss function for TOAI-MR training."""
    
    def __init__(self, vocab_size: int, label_smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        
        # Loss components
        self.translation_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.format_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()
        self.validation_loss = nn.BCELoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss."""
        # Translation loss
        logits = outputs["logits"].view(-1, self.vocab_size)
        target_ids = targets["target_ids"].view(-1)
        trans_loss = self.translation_loss(logits, target_ids)
        
        # Format consistency loss
        format_logits = outputs["format_logits"].view(-1, 10)
        format_targets = targets.get("format_targets", torch.zeros_like(format_logits[:, 0], dtype=torch.long))
        format_loss = self.format_loss(format_logits, format_targets)
        
        # Confidence loss (encourage high confidence for correct translations)
        confidence_scores = outputs["confidence_scores"].view(-1)
        confidence_targets = targets.get("confidence_targets", torch.ones_like(confidence_scores))
        conf_loss = self.confidence_loss(confidence_scores, confidence_targets)
        
        # Medical validation loss
        validation_scores = outputs["validation_scores"].view(-1)
        validation_targets = targets.get("validation_targets", torch.ones_like(validation_scores))
        val_loss = self.validation_loss(validation_scores, validation_targets)
        
        # Combine losses
        total_loss = trans_loss + 0.1 * format_loss + 0.1 * conf_loss + 0.1 * val_loss
        
        return {
            "total_loss": total_loss,
            "translation_loss": trans_loss,
            "format_loss": format_loss,
            "confidence_loss": conf_loss,
            "validation_loss": val_loss
        }

# Configuration for TOAI-MR model
TOAI_MR_CONFIG = {
    "vocab_size": 50000,
    "embed_dim": 1024,
    "num_encoder_layers": 24,
    "num_decoder_layers": 24,
    "num_heads": 16,
    "ff_dim": 4096,
    "max_position_embeddings": 512,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "model_type": "toai_mr"
}

if __name__ == "__main__":
    # Initialize model
    model = TOAIMRModel(TOAI_MR_CONFIG)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"TOAI-MR Model initialized successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**3:.2f} GB (FP32)")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    
    source_ids = torch.randint(0, 1000, (batch_size, seq_length))
    target_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    outputs = model(source_ids, target_ids)
    print(f"Forward pass successful! Output shape: {outputs['logits'].shape}")

