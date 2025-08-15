"""
TOAI-MR Evaluation Framework

This module provides comprehensive evaluation metrics and benchmarking
capabilities for assessing TOAI-MR's performance in medical record translation.
"""

import torch
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from toai_mr_model import TOAIMRModel
from training_pipeline import EHRTranslationDataset

logger = logging.getLogger(__name__)

class MedicalEntityExtractor:
    """Extract and validate medical entities from EHR text."""
    
    def __init__(self):
        """Initialize medical entity patterns."""
        self.entity_patterns = {
            'icd_codes': r'[A-Z]\d{2}\.?\d*',
            'medications': r'(?:MEDICATION:|med:)\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|ml))',
            'lab_values': r'(?:LAB:|lab:)\s*([A-Za-z\s]+)\s*(\d+(?:\.\d+)?)\s*([A-Za-z/]+)',
            'patient_ids': r'(?:MRN:|ID:|PATIENT_ID:)\s*(\d+)',
            'dates': r'\d{4}-\d{2}-\d{2}',
            'procedures': r'(?:PROCEDURE:|proc:)\s*(\d{5})\s*([A-Za-z\s,]+)'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        
        return entities
    
    def compare_entities(self, pred_text: str, ref_text: str) -> Dict[str, float]:
        """Compare medical entities between prediction and reference."""
        pred_entities = self.extract_entities(pred_text)
        ref_entities = self.extract_entities(ref_text)
        
        entity_scores = {}
        
        for entity_type in self.entity_patterns.keys():
            pred_set = set(str(e) for e in pred_entities.get(entity_type, []))
            ref_set = set(str(e) for e in ref_entities.get(entity_type, []))
            
            if len(ref_set) == 0:
                entity_scores[entity_type] = 1.0 if len(pred_set) == 0 else 0.0
            else:
                intersection = len(pred_set.intersection(ref_set))
                entity_scores[entity_type] = intersection / len(ref_set)
        
        return entity_scores

class BLEUScorer:
    """BLEU score computation for translation quality assessment."""
    
    def __init__(self, max_n: int = 4):
        """Initialize BLEU scorer."""
        self.max_n = max_n
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores."""
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        bleu_scores = []
        individual_scores = []
        
        for pred, ref in zip(predictions, references):
            score = self._sentence_bleu(pred, ref)
            bleu_scores.append(score)
            individual_scores.append(score)
        
        return {
            "bleu_score": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "individual_scores": individual_scores
        }
    
    def _sentence_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score for a single sentence pair."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        # Compute brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        # Compute geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean(np.log(precisions)))
        else:
            geo_mean = 0.0
        
        return bp * geo_mean
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-gram counts."""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return dict(ngrams)
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """Compute brevity penalty."""
        if pred_len >= ref_len:
            return 1.0
        else:
            return np.exp(1 - ref_len / pred_len)

class SemanticSimilarityEvaluator:
    """Evaluate semantic similarity between medical texts."""
    
    def __init__(self):
        """Initialize semantic similarity evaluator."""
        # In a real implementation, this would use medical embeddings
        # For demonstration, we'll use a simplified approach
        self.medical_terms_weight = 2.0
        self.medical_terms = {
            'diagnosis', 'medication', 'patient', 'treatment', 'procedure',
            'lab', 'test', 'result', 'condition', 'symptom', 'disease'
        }
    
    def compute_similarity(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute semantic similarity scores."""
        similarities = []
        
        for pred, ref in zip(predictions, references):
            sim = self._text_similarity(pred, ref)
            similarities.append(sim)
        
        return {
            "semantic_similarity": np.mean(similarities),
            "similarity_std": np.std(similarities),
            "individual_similarities": similarities
        }
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Basic Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Boost similarity for medical terms
        medical_intersection = len(tokens1.intersection(tokens2).intersection(self.medical_terms))
        medical_boost = medical_intersection * 0.1
        
        return min(1.0, jaccard_sim + medical_boost)

class ClinicalValidationEvaluator:
    """Evaluate clinical validity of translations."""
    
    def __init__(self):
        """Initialize clinical validation rules."""
        self.validation_rules = {
            'patient_consistency': self._check_patient_consistency,
            'date_consistency': self._check_date_consistency,
            'medication_dosage': self._check_medication_dosage,
            'lab_ranges': self._check_lab_ranges,
            'icd_validity': self._check_icd_validity
        }
    
    def validate_translations(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Validate clinical accuracy of translations."""
        validation_results = {rule: [] for rule in self.validation_rules}
        
        for pred, ref in zip(predictions, references):
            for rule_name, rule_func in self.validation_rules.items():
                result = rule_func(pred, ref)
                validation_results[rule_name].append(result)
        
        # Compute aggregate scores
        aggregate_results = {}
        for rule_name, results in validation_results.items():
            aggregate_results[rule_name] = {
                'accuracy': np.mean(results),
                'total_checks': len(results),
                'passed_checks': sum(results)
            }
        
        # Overall clinical validity score
        overall_score = np.mean([np.mean(results) for results in validation_results.values()])
        aggregate_results['overall_clinical_validity'] = overall_score
        
        return aggregate_results
    
    def _check_patient_consistency(self, pred: str, ref: str) -> bool:
        """Check if patient information is consistent."""
        # Extract patient IDs
        pred_ids = re.findall(r'(?:MRN:|ID:)\s*(\d+)', pred)
        ref_ids = re.findall(r'(?:MRN:|ID:)\s*(\d+)', ref)
        
        if not pred_ids or not ref_ids:
            return True  # No patient IDs to compare
        
        return pred_ids[0] == ref_ids[0]
    
    def _check_date_consistency(self, pred: str, ref: str) -> bool:
        """Check if dates are consistent."""
        pred_dates = re.findall(r'\d{4}-\d{2}-\d{2}', pred)
        ref_dates = re.findall(r'\d{4}-\d{2}-\d{2}', ref)
        
        if not pred_dates or not ref_dates:
            return True  # No dates to compare
        
        return set(pred_dates) == set(ref_dates)
    
    def _check_medication_dosage(self, pred: str, ref: str) -> bool:
        """Check if medication dosages are reasonable."""
        # Extract medication dosages
        pred_meds = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|mcg)', pred.lower())
        ref_meds = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|mcg)', ref.lower())
        
        if not pred_meds or not ref_meds:
            return True
        
        # Check if dosages are within reasonable ranges
        for dose, unit in pred_meds:
            dose_val = float(dose)
            if unit == 'mg' and (dose_val < 0.1 or dose_val > 5000):
                return False
            elif unit == 'mcg' and (dose_val < 1 or dose_val > 10000):
                return False
        
        return True
    
    def _check_lab_ranges(self, pred: str, ref: str) -> bool:
        """Check if lab values are within reasonable ranges."""
        # Extract lab values (simplified)
        pred_labs = re.findall(r'(\d+(?:\.\d+)?)', pred)
        ref_labs = re.findall(r'(\d+(?:\.\d+)?)', ref)
        
        if not pred_labs:
            return True
        
        # Basic sanity check for lab values
        for lab_val in pred_labs:
            val = float(lab_val)
            if val < 0 or val > 10000:  # Very broad range
                return False
        
        return True
    
    def _check_icd_validity(self, pred: str, ref: str) -> bool:
        """Check if ICD codes follow valid format."""
        pred_codes = re.findall(r'[A-Z]\d{2}\.?\d*', pred)
        
        if not pred_codes:
            return True
        
        # Check ICD-10 format
        for code in pred_codes:
            if not re.match(r'^[A-Z]\d{2}(\.\d+)?$', code):
                return False
        
        return True

class TOAIMREvaluator:
    """Comprehensive evaluator for TOAI-MR model."""
    
    def __init__(self, model_path: str):
        """Initialize evaluator with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TOAIMRModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Initialize evaluation components
        self.entity_extractor = MedicalEntityExtractor()
        self.bleu_scorer = BLEUScorer()
        self.semantic_evaluator = SemanticSimilarityEvaluator()
        self.clinical_validator = ClinicalValidationEvaluator()
        
        logger.info(f"TOAI-MR Evaluator initialized with model from {model_path}")
    
    def evaluate_dataset(self, test_file: str, output_file: str) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info(f"Evaluating on dataset: {test_file}")
        
        # Load test data
        test_dataset = EHRTranslationDataset(test_file, self.model.tokenizer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        predictions = []
        references = []
        confidence_scores = []
        translation_times = []
        
        # Generate predictions
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}/{len(test_loader)}")
                
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Time translation
                start_time = datetime.now()
                
                # Generate translation
                generated = self.model.generate(
                    batch["source_input_ids"],
                    batch["source_attention_mask"],
                    max_length=512
                )
                
                end_time = datetime.now()
                translation_time = (end_time - start_time).total_seconds()
                
                # Store results
                predictions.extend(generated["generated_text"])
                confidence_scores.extend(generated["confidence_scores"].cpu().numpy())
                translation_times.append(translation_time)
                
                # Get reference
                ref_text = self.model.tokenizer.decode(batch["target_input_ids"].squeeze())
                references.append(ref_text)
        
        # Compute evaluation metrics
        results = self._compute_all_metrics(predictions, references, confidence_scores, translation_times)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_file}")
        return results
    
    def _compute_all_metrics(self, predictions: List[str], references: List[str], 
                           confidence_scores: List[float], translation_times: List[float]) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "model_info": {
                "model_type": "TOAI-MR",
                "parameters": sum(p.numel() for p in self.model.parameters())
            }
        }
        
        # BLEU scores
        logger.info("Computing BLEU scores...")
        bleu_results = self.bleu_scorer.compute_bleu(predictions, references)
        results["bleu_metrics"] = bleu_results
        
        # Semantic similarity
        logger.info("Computing semantic similarity...")
        semantic_results = self.semantic_evaluator.compute_similarity(predictions, references)
        results["semantic_metrics"] = semantic_results
        
        # Medical entity accuracy
        logger.info("Computing medical entity accuracy...")
        entity_results = self._compute_entity_metrics(predictions, references)
        results["entity_metrics"] = entity_results
        
        # Clinical validation
        logger.info("Computing clinical validation...")
        clinical_results = self.clinical_validator.validate_translations(predictions, references)
        results["clinical_metrics"] = clinical_results
        
        # Performance metrics
        results["performance_metrics"] = {
            "avg_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "avg_translation_time": np.mean(translation_times),
            "translation_time_std": np.std(translation_times),
            "translations_per_second": 1.0 / np.mean(translation_times)
        }
        
        # Overall score
        overall_score = self._compute_overall_score(results)
        results["overall_score"] = overall_score
        
        return results
    
    def _compute_entity_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute medical entity extraction metrics."""
        entity_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.entity_extractor.compare_entities(pred, ref)
            for entity_type, score in scores.items():
                entity_scores[entity_type].append(score)
        
        # Aggregate results
        entity_results = {}
        for entity_type, scores in entity_scores.items():
            entity_results[entity_type] = {
                "accuracy": np.mean(scores),
                "std": np.std(scores),
                "samples": len(scores)
            }
        
        # Overall entity accuracy
        all_scores = [score for scores in entity_scores.values() for score in scores]
        entity_results["overall_entity_accuracy"] = np.mean(all_scores) if all_scores else 0.0
        
        return entity_results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute weighted overall performance score."""
        weights = {
            "bleu_score": 0.3,
            "semantic_similarity": 0.25,
            "overall_entity_accuracy": 0.25,
            "overall_clinical_validity": 0.2
        }
        
        score_components = {
            "bleu_score": results["bleu_metrics"]["bleu_score"],
            "semantic_similarity": results["semantic_metrics"]["semantic_similarity"],
            "overall_entity_accuracy": results["entity_metrics"]["overall_entity_accuracy"],
            "overall_clinical_validity": results["clinical_metrics"]["overall_clinical_validity"]
        }
        
        overall_score = sum(weights[key] * score_components[key] for key in weights.keys())
        return overall_score
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_file: str) -> None:
        """Generate comprehensive evaluation report."""
        report_lines = [
            "# TOAI-MR Evaluation Report",
            f"Generated on: {results['evaluation_timestamp']}",
            f"Number of samples: {results['num_samples']}",
            "",
            "## Overall Performance",
            f"Overall Score: {results['overall_score']:.4f}",
            "",
            "## Translation Quality Metrics",
            f"BLEU Score: {results['bleu_metrics']['bleu_score']:.4f} ± {results['bleu_metrics']['bleu_std']:.4f}",
            f"Semantic Similarity: {results['semantic_metrics']['semantic_similarity']:.4f} ± {results['semantic_metrics']['similarity_std']:.4f}",
            "",
            "## Medical Entity Accuracy",
            f"Overall Entity Accuracy: {results['entity_metrics']['overall_entity_accuracy']:.4f}",
            ""
        ]
        
        # Add entity-specific results
        for entity_type, metrics in results['entity_metrics'].items():
            if entity_type != 'overall_entity_accuracy':
                report_lines.append(f"{entity_type}: {metrics['accuracy']:.4f} ± {metrics['std']:.4f}")
        
        report_lines.extend([
            "",
            "## Clinical Validation",
            f"Overall Clinical Validity: {results['clinical_metrics']['overall_clinical_validity']:.4f}",
            ""
        ])
        
        # Add clinical validation details
        for rule_name, metrics in results['clinical_metrics'].items():
            if rule_name != 'overall_clinical_validity':
                report_lines.append(f"{rule_name}: {metrics['accuracy']:.4f} ({metrics['passed_checks']}/{metrics['total_checks']})")
        
        report_lines.extend([
            "",
            "## Performance Metrics",
            f"Average Confidence: {results['performance_metrics']['avg_confidence']:.4f}",
            f"Average Translation Time: {results['performance_metrics']['avg_translation_time']:.4f}s",
            f"Translations per Second: {results['performance_metrics']['translations_per_second']:.2f}",
        ])
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Evaluation report saved to {output_file}")

def main():
    """Main evaluation function."""
    # Configuration
    model_path = "/home/ubuntu/toai_mr_code/models/toai_mr_v1/best_model"
    test_file = "/home/ubuntu/toai_mr_code/data/test_dataset.json"
    results_file = "/home/ubuntu/toai_mr_code/evaluation_results.json"
    report_file = "/home/ubuntu/toai_mr_code/evaluation_report.md"
    
    try:
        # Initialize evaluator
        evaluator = TOAIMREvaluator(model_path)
        
        # Run evaluation
        results = evaluator.evaluate_dataset(test_file, results_file)
        
        # Generate report
        evaluator.generate_evaluation_report(results, report_file)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Overall Score: {results['overall_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

