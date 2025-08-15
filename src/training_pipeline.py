import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from datetime import datetime

from toai_mr_model import TOAIMRModel, TOAIMRLoss, TOAI_MR_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EHRTranslationDataset(Dataset):
    """Dataset class for EHR translation training data."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        """Initialize dataset."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        sample = self.data[idx]
        
        # Convert records to text
        source_text = self._record_to_text(sample["source_record"], sample["source_format"])
        target_text = self._record_to_text(sample["target_record"], sample["target_format"])
        
        # Tokenize
        source_encoding = self.tokenizer.encode(source_text, max_length=self.max_length)
        target_encoding = self.tokenizer.encode(target_text, max_length=self.max_length)
        
        return {
            "source_input_ids": source_encoding["input_ids"].squeeze(),
            "source_attention_mask": source_encoding["attention_mask"].squeeze(),
            "target_input_ids": target_encoding["input_ids"].squeeze(),
            "target_attention_mask": target_encoding["attention_mask"].squeeze(),
            "source_format": sample["source_format"],
            "target_format": sample["target_format"]
        }
    
    def _record_to_text(self, record: Dict[str, Any], format_type: str) -> str:
        """Convert EHR record to text representation."""
        if format_type == "epic":
            return self._epic_to_text(record["epic_record"])
        elif format_type == "cerner":
            return self._cerner_to_text(record["cerner_record"])
        elif format_type == "fhir":
            return self._fhir_to_text(record)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _epic_to_text(self, record: Dict[str, Any]) -> str:
        """Convert Epic record to text."""
        text_parts = []
        
        # Patient info
        patient = record["patient_info"]
        text_parts.append(f"EPIC_PATIENT: {patient['patient_name']}")
        text_parts.append(f"MRN: {patient['epic_mrn']}")
        text_parts.append(f"DOB: {patient['dob']}")
        text_parts.append(f"SEX: {patient['sex']}")
        
        # Encounter
        encounter = record["encounter_data"]
        text_parts.append(f"ENCOUNTER: {encounter['csn']}")
        text_parts.append(f"DATE: {encounter['encounter_date']}")
        text_parts.append(f"CHIEF_COMPLAINT: {encounter['chief_complaint']}")
        
        # Diagnoses
        for dx in record["diagnosis_list"]:
            text_parts.append(f"DIAGNOSIS: {dx['icd10_code']} {dx['diagnosis_name']}")
        
        # Medications
        for med in record["medication_list"]:
            text_parts.append(f"MEDICATION: {med['medication_name']} {med['dose']} {med['frequency']}")
        
        # Lab results
        for lab in record["lab_results"]:
            text_parts.append(f"LAB: {lab['component_name']} {lab['result_value']} {lab['units']}")
        
        return " | ".join(text_parts)
    
    def _cerner_to_text(self, record: Dict[str, Any]) -> str:
        """Convert Cerner record to text."""
        text_parts = []
        
        # Person info
        person = record["person"]
        text_parts.append(f"CERNER_PERSON: {person['name_full_formatted']}")
        text_parts.append(f"PERSON_ID: {person['person_id']}")
        text_parts.append(f"BIRTH_DT: {person['birth_dt_tm']}")
        text_parts.append(f"SEX: {person['sex_cd']}")
        
        # Encounter
        encounter = record["encntr"]
        text_parts.append(f"ENCOUNTER: {encounter['encntr_id']}")
        text_parts.append(f"ARRIVE_DT: {encounter['arrive_dt_tm']}")
        text_parts.append(f"REASON: {encounter['reason_for_visit']}")
        
        # Diagnoses
        for dx in record["diagnosis"]:
            text_parts.append(f"DIAGNOSIS: {dx['nomenclature_id']} {dx['source_string']}")
        
        # Orders
        for order in record["orders"]:
            text_parts.append(f"ORDER: {order['catalog_cd']} {order['ordered_as_mnemonic']}")
        
        # Clinical events
        for event in record["clinical_event"]:
            text_parts.append(f"EVENT: {event['event_cd']} {event['result_val']} {event['result_units_cd']}")
        
        return " | ".join(text_parts)
    
    def _fhir_to_text(self, record: Dict[str, Any]) -> str:
        """Convert FHIR record to text."""
        text_parts = []
        
        for entry in record["entry"]:
            resource = entry["resource"]
            resource_type = resource["resourceType"]
            
            if resource_type == "Patient":
                name = resource["name"][0]
                text_parts.append(f"FHIR_PATIENT: {name['given'][0]} {name['family']}")
                text_parts.append(f"ID: {resource['id']}")
                text_parts.append(f"BIRTH_DATE: {resource['birthDate']}")
                text_parts.append(f"GENDER: {resource['gender']}")
            
            elif resource_type == "Encounter":
                text_parts.append(f"ENCOUNTER: {resource['id']}")
                text_parts.append(f"STATUS: {resource['status']}")
                if "reasonCode" in resource:
                    text_parts.append(f"REASON: {resource['reasonCode'][0]['text']}")
            
            elif resource_type == "Condition":
                code = resource["code"]["coding"][0]
                text_parts.append(f"CONDITION: {code['code']} {code['display']}")
            
            elif resource_type == "MedicationRequest":
                med_text = resource["medicationCodeableConcept"]["text"]
                dosage = resource["dosageInstruction"][0]["text"]
                text_parts.append(f"MEDICATION: {med_text} {dosage}")
            
            elif resource_type == "Observation":
                test_name = resource["code"]["text"]
                value = resource["valueQuantity"]
                text_parts.append(f"OBSERVATION: {test_name} {value['value']} {value['unit']}")
        
        return " | ".join(text_parts)

class TOAIMRTrainer:
    """Training class for TOAI-MR model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = TOAIMRModel(TOAI_MR_CONFIG).to(self.device)
        
        # Initialize loss function
        self.loss_fn = TOAIMRLoss(TOAI_MR_CONFIG["vocab_size"])
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, train_file: str, val_file: str, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Create datasets
        train_dataset = EHRTranslationDataset(train_file, self.model.tokenizer)
        val_dataset = EHRTranslationDataset(val_file, self.model.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize scheduler with total steps
        total_steps = len(train_loader) * self.config.get("num_epochs", 10)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Training batches: {len(train_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                batch["source_input_ids"],
                batch["target_input_ids"],
                batch["source_attention_mask"],
                batch["target_attention_mask"]
            )
            
            # Prepare targets
            targets = {
                "target_ids": batch["target_input_ids"],
                "confidence_targets": torch.ones_like(outputs["confidence_scores"].view(-1)),
                "validation_targets": torch.ones_like(outputs["validation_scores"].view(-1))
            }
            
            # Compute loss
            loss_dict = self.loss_fn(outputs, targets)
            loss = loss_dict["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb if available
            if hasattr(self, 'wandb_run'):
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.global_step
                })
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return {
            "train_loss": avg_loss,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch["source_input_ids"],
                    batch["target_input_ids"],
                    batch["source_attention_mask"],
                    batch["target_attention_mask"]
                )
                
                # Prepare targets
                targets = {
                    "target_ids": batch["target_input_ids"],
                    "confidence_targets": torch.ones_like(outputs["confidence_scores"].view(-1)),
                    "validation_targets": torch.ones_like(outputs["validation_scores"].view(-1))
                }
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, targets)
                val_losses.append(loss_dict["total_loss"].item())
        
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        return {"val_loss": avg_val_loss}
    
    def evaluate_translation_quality(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate translation quality with medical-specific metrics."""
        self.model.eval()
        
        predictions = []
        references = []
        confidence_scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate translations
                generated = self.model.generate(
                    batch["source_input_ids"],
                    batch["source_attention_mask"],
                    max_length=512
                )
                
                predictions.extend(generated["generated_text"])
                confidence_scores.extend(generated["confidence_scores"].cpu().numpy())
                
                # Get reference translations
                for target_ids in batch["target_input_ids"]:
                    ref_text = self.model.tokenizer.decode(target_ids)
                    references.append(ref_text)
        
        # Compute BLEU score (simplified)
        bleu_score = self._compute_bleu_score(predictions, references)
        
        # Compute medical entity accuracy (simplified)
        entity_accuracy = self._compute_entity_accuracy(predictions, references)
        
        # Average confidence
        avg_confidence = np.mean(confidence_scores)
        
        return {
            "bleu_score": bleu_score,
            "entity_accuracy": entity_accuracy,
            "avg_confidence": avg_confidence,
            "num_samples": len(predictions)
        }
    
    def _compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute simplified BLEU score."""
        # Simplified BLEU computation for demonstration
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if len(ref_words) > 0:
                overlap = len(pred_words.intersection(ref_words))
                score = overlap / len(ref_words)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_entity_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute medical entity accuracy."""
        # Simplified entity accuracy for demonstration
        medical_entities = ["DIAGNOSIS:", "MEDICATION:", "LAB:", "PATIENT:", "ENCOUNTER:"]
        
        scores = []
        for pred, ref in zip(predictions, references):
            pred_entities = [entity for entity in medical_entities if entity in pred]
            ref_entities = [entity for entity in medical_entities if entity in ref]
            
            if len(ref_entities) > 0:
                overlap = len(set(pred_entities).intersection(set(ref_entities)))
                score = overlap / len(ref_entities)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, train_file: str, val_file: str, output_dir: str) -> None:
        """Complete training pipeline."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_file, val_file, 
            batch_size=self.config.get("batch_size", 8)
        )
        
        # Training loop
        num_epochs = self.config.get("num_epochs", 10)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(output_dir, is_best=is_best)
            
            # Log to wandb if available
            if hasattr(self, 'wandb_run'):
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_metrics['val_loss'],
                    "best_val_loss": self.best_val_loss
                })
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

def main():
    """Main training function."""
    # Training configuration
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 4,  # Reduced for memory constraints
        "num_epochs": 5,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
        "save_steps": 1000,
        "eval_steps": 500,
        "logging_steps": 100
    }
    
    # Initialize trainer
    trainer = TOAIMRTrainer(config)
    
    # Data files
    train_file = "/home/ubuntu/toai_mr_code/data/training_dataset.json"
    val_file = "/home/ubuntu/toai_mr_code/data/validation_dataset.json"
    output_dir = "/home/ubuntu/toai_mr_code/models/toai_mr_v1"
    
    # Start training
    trainer.train(train_file, val_file, output_dir)

if __name__ == "__main__":
    main()

