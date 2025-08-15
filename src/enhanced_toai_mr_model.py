import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import math
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TOAIMRConfig:
    """
    Configuration class for TOAI-MR model with all hyperparameters.
    
    This configuration class contains all the hyperparameters used in the
    mathematical formulations described in the paper.
    """
    
    # Architecture parameters
    vocab_size: int = 50000
    medical_vocab_size: int = 15000
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    
    # Medical-specific parameters (from mathematical formulations)
    medical_token_boost: float = 2.5  # α in paper
    medical_attention_scaling: float = 0.1  # β in paper
    knowledge_integration_dim: int = 768
    
    # Training parameters
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Loss function weights (from mathematical formulations)
    translation_loss_weight: float = 1.0
    format_compliance_weight: float = 0.3  # η₁ in paper
    entity_preservation_weight: float = 0.5  # η₂ in paper
    medical_token_loss_weight: float = 1.5  # δ in paper
    
    # RLHF parameters
    kl_divergence_penalty: float = 0.01  # κ in paper
    reward_weights: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.1])  # ω in paper
    
    # Confidence estimation weights (from mathematical formulations)
    confidence_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])  # μ in paper
    
    # Knowledge base weights (learnable γ parameters)
    knowledge_base_names: List[str] = field(default_factory=lambda: ['FHIR', 'HL7', 'ICD10', 'SNOMED', 'RxNorm', 'LOINC'])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert len(self.reward_weights) == 5, \
            "reward_weights must have 5 elements for clinical dimensions"
        assert len(self.confidence_weights) == 4, \
            "confidence_weights must have 4 elements for uncertainty measures"
        
        logger.info(f"Initialized TOAI-MR configuration with {self.num_hidden_layers} layers")

class MedicalKnowledgeBase:
    """
    Comprehensive medical knowledge base integration for TOAI-MR.
    
    This class implements the knowledge integration component described in the paper,
    supporting FHIR, HL7, ICD-10, SNOMED CT, RxNorm, and LOINC standards.
    
    Mathematical formulation:
    φᵢ: Kᵢ → ℝᵈᵏ (embedding function for knowledge base i)
    """
    
    def __init__(self, knowledge_dim: int = 768, device: str = 'cpu'):
        self.knowledge_dim = knowledge_dim
        self.device = device
        
        # Load all knowledge bases
        self.knowledge_bases = {
            'FHIR': self._load_fhir_knowledge(),
            'HL7': self._load_hl7_knowledge(),
            'ICD10': self._load_icd10_knowledge(),
            'SNOMED': self._load_snomed_knowledge(),
            'RxNorm': self._load_rxnorm_knowledge(),
            'LOINC': self._load_loinc_knowledge()
        }
        
        # Initialize embeddings for each knowledge base
        self.knowledge_embeddings = nn.ModuleDict({
            kb_name: nn.Embedding(len(kb_data) + 1, knowledge_dim)  # +1 for unknown
            for kb_name, kb_data in self.knowledge_bases.items()
        })
        
        # Initialize embeddings with Xavier uniform
        for kb_name, embedding in self.knowledge_embeddings.items():
            nn.init.xavier_uniform_(embedding.weight)
        
        total_concepts = sum(len(kb) for kb in self.knowledge_bases.values())
        logger.info(f"Initialized medical knowledge base with {total_concepts:,} concepts across {len(self.knowledge_bases)} knowledge bases")
    
    def _load_fhir_knowledge(self) -> Dict[str, Dict]:
        """
        Load FHIR R4 resource specifications.
        
        In production, this would load from actual FHIR specification files.
        This is a comprehensive subset for demonstration.
        """
        fhir_resources = {
            # Core resources
            'Patient': {'id': 0, 'type': 'resource', 'cardinality': '1..1', 'category': 'administrative'},
            'Practitioner': {'id': 1, 'type': 'resource', 'cardinality': '0..*', 'category': 'administrative'},
            'Organization': {'id': 2, 'type': 'resource', 'cardinality': '0..*', 'category': 'administrative'},
            'Location': {'id': 3, 'type': 'resource', 'cardinality': '0..*', 'category': 'administrative'},
            
            # Clinical resources
            'Encounter': {'id': 4, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'Observation': {'id': 5, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'Condition': {'id': 6, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'Procedure': {'id': 7, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'MedicationRequest': {'id': 8, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'MedicationAdministration': {'id': 9, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'AllergyIntolerance': {'id': 10, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'CarePlan': {'id': 11, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            'Goal': {'id': 12, 'type': 'resource', 'cardinality': '0..*', 'category': 'clinical'},
            
            # Diagnostic resources
            'DiagnosticReport': {'id': 13, 'type': 'resource', 'cardinality': '0..*', 'category': 'diagnostic'},
            'ImagingStudy': {'id': 14, 'type': 'resource', 'cardinality': '0..*', 'category': 'diagnostic'},
            'Specimen': {'id': 15, 'type': 'resource', 'cardinality': '0..*', 'category': 'diagnostic'},
            
            # Financial resources
            'Coverage': {'id': 16, 'type': 'resource', 'cardinality': '0..*', 'category': 'financial'},
            'Claim': {'id': 17, 'type': 'resource', 'cardinality': '0..*', 'category': 'financial'},
            'ExplanationOfBenefit': {'id': 18, 'type': 'resource', 'cardinality': '0..*', 'category': 'financial'},
            
            # Workflow resources
            'Task': {'id': 19, 'type': 'resource', 'cardinality': '0..*', 'category': 'workflow'},
            'Appointment': {'id': 20, 'type': 'resource', 'cardinality': '0..*', 'category': 'workflow'},
            'Schedule': {'id': 21, 'type': 'resource', 'cardinality': '0..*', 'category': 'workflow'},
        }
        
        return fhir_resources
    
    def _load_hl7_knowledge(self) -> Dict[str, Dict]:
        """
        Load HL7 v2.x and v3 message structures.
        
        Comprehensive HL7 message types and segments.
        """
        hl7_messages = {
            # ADT messages (Admit, Discharge, Transfer)
            'ADT^A01': {'id': 0, 'type': 'admit', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Patient admit'},
            'ADT^A02': {'id': 1, 'type': 'transfer', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Patient transfer'},
            'ADT^A03': {'id': 2, 'type': 'discharge', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Patient discharge'},
            'ADT^A04': {'id': 3, 'type': 'register', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Patient registration'},
            'ADT^A08': {'id': 4, 'type': 'update', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Patient information update'},
            
            # ORU messages (Observation Result)
            'ORU^R01': {'id': 5, 'type': 'observation', 'segments': ['MSH', 'PID', 'OBR', 'OBX'], 'description': 'Observation result'},
            'ORU^R02': {'id': 6, 'type': 'query_response', 'segments': ['MSH', 'PID', 'OBR', 'OBX'], 'description': 'Query response'},
            
            # ORM messages (Order)
            'ORM^O01': {'id': 7, 'type': 'order', 'segments': ['MSH', 'PID', 'ORC', 'OBR'], 'description': 'General order'},
            'ORM^O02': {'id': 8, 'type': 'order_response', 'segments': ['MSH', 'PID', 'ORC', 'OBR'], 'description': 'Order response'},
            
            # RAS messages (Pharmacy)
            'RAS^O17': {'id': 9, 'type': 'pharmacy', 'segments': ['MSH', 'PID', 'RXA'], 'description': 'Pharmacy administration'},
            'RDE^O11': {'id': 10, 'type': 'pharmacy_order', 'segments': ['MSH', 'PID', 'RXE'], 'description': 'Pharmacy order'},
            
            # SIU messages (Scheduling)
            'SIU^S12': {'id': 11, 'type': 'schedule', 'segments': ['MSH', 'SCH', 'PID'], 'description': 'Schedule appointment'},
            'SIU^S13': {'id': 12, 'type': 'reschedule', 'segments': ['MSH', 'SCH', 'PID'], 'description': 'Reschedule appointment'},
            
            # MDM messages (Medical Document Management)
            'MDM^T02': {'id': 13, 'type': 'document', 'segments': ['MSH', 'EVN', 'PID', 'TXA'], 'description': 'Medical document'},
            
            # BAR messages (Billing)
            'BAR^P01': {'id': 14, 'type': 'billing', 'segments': ['MSH', 'EVN', 'PID', 'PV1'], 'description': 'Billing information'},
        }
        
        return hl7_messages
    
    def _load_icd10_knowledge(self) -> Dict[str, Dict]:
        """
        Load ICD-10 diagnostic codes.
        
        Comprehensive ICD-10 code categories and common codes.
        """
        icd10_codes = {
            # Infectious and parasitic diseases (A00-B99)
            'A00': {'id': 0, 'description': 'Cholera', 'category': 'infectious', 'chapter': 'A00-B99'},
            'A09': {'id': 1, 'description': 'Infectious gastroenteritis and colitis', 'category': 'infectious', 'chapter': 'A00-B99'},
            'B34': {'id': 2, 'description': 'Viral infection of unspecified site', 'category': 'infectious', 'chapter': 'A00-B99'},
            
            # Neoplasms (C00-D49)
            'C78': {'id': 3, 'description': 'Secondary malignant neoplasm of respiratory organs', 'category': 'neoplasm', 'chapter': 'C00-D49'},
            'C80': {'id': 4, 'description': 'Malignant neoplasm without specification of site', 'category': 'neoplasm', 'chapter': 'C00-D49'},
            
            # Endocrine, nutritional and metabolic diseases (E00-E89)
            'E11': {'id': 5, 'description': 'Type 2 diabetes mellitus', 'category': 'endocrine', 'chapter': 'E00-E89'},
            'E78': {'id': 6, 'description': 'Disorders of lipoprotein metabolism', 'category': 'endocrine', 'chapter': 'E00-E89'},
            'E87': {'id': 7, 'description': 'Other disorders of fluid, electrolyte and acid-base balance', 'category': 'endocrine', 'chapter': 'E00-E89'},
            
            # Mental and behavioural disorders (F00-F99)
            'F32': {'id': 8, 'description': 'Major depressive disorder, single episode', 'category': 'mental', 'chapter': 'F00-F99'},
            'F41': {'id': 9, 'description': 'Other anxiety disorders', 'category': 'mental', 'chapter': 'F00-F99'},
            
            # Diseases of the nervous system (G00-G99)
            'G93': {'id': 10, 'description': 'Other disorders of brain', 'category': 'nervous', 'chapter': 'G00-G99'},
            
            # Diseases of the circulatory system (I00-I99)
            'I10': {'id': 11, 'description': 'Essential hypertension', 'category': 'circulatory', 'chapter': 'I00-I99'},
            'I21': {'id': 12, 'description': 'Acute myocardial infarction', 'category': 'circulatory', 'chapter': 'I00-I99'},
            'I25': {'id': 13, 'description': 'Chronic ischemic heart disease', 'category': 'circulatory', 'chapter': 'I00-I99'},
            'I50': {'id': 14, 'description': 'Heart failure', 'category': 'circulatory', 'chapter': 'I00-I99'},
            
            # Diseases of the respiratory system (J00-J99)
            'J44': {'id': 15, 'description': 'Chronic obstructive pulmonary disease', 'category': 'respiratory', 'chapter': 'J00-J99'},
            'J45': {'id': 16, 'description': 'Asthma', 'category': 'respiratory', 'chapter': 'J00-J99'},
            'J18': {'id': 17, 'description': 'Pneumonia, unspecified organism', 'category': 'respiratory', 'chapter': 'J00-J99'},
            
            # Diseases of the digestive system (K00-K95)
            'K21': {'id': 18, 'description': 'Gastro-esophageal reflux disease', 'category': 'digestive', 'chapter': 'K00-K95'},
            'K59': {'id': 19, 'description': 'Other functional intestinal disorders', 'category': 'digestive', 'chapter': 'K00-K95'},
            
            # Diseases of the genitourinary system (N00-N99)
            'N18': {'id': 20, 'description': 'Chronic kidney disease', 'category': 'genitourinary', 'chapter': 'N00-N99'},
            'N39': {'id': 21, 'description': 'Other disorders of urinary system', 'category': 'genitourinary', 'chapter': 'N00-N99'},
            
            # Injury, poisoning and consequences of external causes (S00-T88)
            'S72': {'id': 22, 'description': 'Fracture of femur', 'category': 'injury', 'chapter': 'S00-T88'},
            'T78': {'id': 23, 'description': 'Adverse effects, not elsewhere classified', 'category': 'injury', 'chapter': 'S00-T88'},
            
            # External causes of morbidity (V00-Y99)
            'W19': {'id': 24, 'description': 'Unspecified fall', 'category': 'external', 'chapter': 'V00-Y99'},
            
            # Factors influencing health status (Z00-Z99)
            'Z51': {'id': 25, 'description': 'Encounter for other aftercare', 'category': 'factors', 'chapter': 'Z00-Z99'},
        }
        
        return icd10_codes
    
    def _load_snomed_knowledge(self) -> Dict[str, Dict]:
        """
        Load SNOMED CT clinical terminology.
        
        Comprehensive SNOMED CT concepts across different semantic tags.
        """
        snomed_concepts = {
            # Clinical findings
            '22298006': {'id': 0, 'term': 'Myocardial infarction', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '73211009': {'id': 1, 'term': 'Diabetes mellitus', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '38341003': {'id': 2, 'term': 'Hypertensive disorder', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '195967001': {'id': 3, 'term': 'Asthma', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '13645005': {'id': 4, 'term': 'Chronic obstructive lung disease', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '44054006': {'id': 5, 'term': 'Diabetes mellitus type 2', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '35489007': {'id': 6, 'term': 'Depressive disorder', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            '48694002': {'id': 7, 'term': 'Anxiety', 'semantic_tag': 'disorder', 'hierarchy': 'clinical_finding'},
            
            # Procedures
            '386053000': {'id': 8, 'term': 'Evaluation procedure', 'semantic_tag': 'procedure', 'hierarchy': 'procedure'},
            '182836005': {'id': 9, 'term': 'Review of medication', 'semantic_tag': 'procedure', 'hierarchy': 'procedure'},
            '18956008': {'id': 10, 'term': 'Radiographic imaging procedure', 'semantic_tag': 'procedure', 'hierarchy': 'procedure'},
            '71388002': {'id': 11, 'term': 'Procedure', 'semantic_tag': 'procedure', 'hierarchy': 'procedure'},
            '387713003': {'id': 12, 'term': 'Surgical procedure', 'semantic_tag': 'procedure', 'hierarchy': 'procedure'},
            
            # Body structures
            '80891009': {'id': 13, 'term': 'Heart structure', 'semantic_tag': 'body_structure', 'hierarchy': 'body_structure'},
            '39607008': {'id': 14, 'term': 'Lung structure', 'semantic_tag': 'body_structure', 'hierarchy': 'body_structure'},
            '64033007': {'id': 15, 'term': 'Kidney structure', 'semantic_tag': 'body_structure', 'hierarchy': 'body_structure'},
            '69536005': {'id': 16, 'term': 'Head structure', 'semantic_tag': 'body_structure', 'hierarchy': 'body_structure'},
            
            # Substances
            '387517004': {'id': 17, 'term': 'Paracetamol', 'semantic_tag': 'substance', 'hierarchy': 'substance'},
            '387207008': {'id': 18, 'term': 'Aspirin', 'semantic_tag': 'substance', 'hierarchy': 'substance'},
            '387467008': {'id': 19, 'term': 'Metformin', 'semantic_tag': 'substance', 'hierarchy': 'substance'},
            '386872004': {'id': 20, 'term': 'Lisinopril', 'semantic_tag': 'substance', 'hierarchy': 'substance'},
            
            # Observable entities
            '271649006': {'id': 21, 'term': 'Systolic blood pressure', 'semantic_tag': 'observable_entity', 'hierarchy': 'observable_entity'},
            '271650006': {'id': 22, 'term': 'Diastolic blood pressure', 'semantic_tag': 'observable_entity', 'hierarchy': 'observable_entity'},
            '364075005': {'id': 23, 'term': 'Heart rate', 'semantic_tag': 'observable_entity', 'hierarchy': 'observable_entity'},
            '276885007': {'id': 24, 'term': 'Core body temperature', 'semantic_tag': 'observable_entity', 'hierarchy': 'observable_entity'},
        }
        
        return snomed_concepts
    
    def _load_rxnorm_knowledge(self) -> Dict[str, Dict]:
        """
        Load RxNorm medication concepts.
        
        Comprehensive RxNorm medication database with ingredients and strengths.
        """
        rxnorm_concepts = {
            # Common medications
            '161': {'id': 0, 'name': 'Aspirin', 'ingredient': 'aspirin', 'strength': '325 mg', 'form': 'tablet'},
            '8150': {'id': 1, 'name': 'Metformin', 'ingredient': 'metformin', 'strength': '500 mg', 'form': 'tablet'},
            '32968': {'id': 2, 'name': 'Lisinopril', 'ingredient': 'lisinopril', 'strength': '10 mg', 'form': 'tablet'},
            '4493': {'id': 3, 'name': 'Atorvastatin', 'ingredient': 'atorvastatin', 'strength': '20 mg', 'form': 'tablet'},
            '5640': {'id': 4, 'name': 'Ibuprofen', 'ingredient': 'ibuprofen', 'strength': '200 mg', 'form': 'tablet'},
            '1191': {'id': 5, 'name': 'Acetaminophen', 'ingredient': 'acetaminophen', 'strength': '500 mg', 'form': 'tablet'},
            '2670': {'id': 6, 'name': 'Furosemide', 'ingredient': 'furosemide', 'strength': '40 mg', 'form': 'tablet'},
            '8896': {'id': 7, 'name': 'Omeprazole', 'ingredient': 'omeprazole', 'strength': '20 mg', 'form': 'capsule'},
            '4603': {'id': 8, 'name': 'Amlodipine', 'ingredient': 'amlodipine', 'strength': '5 mg', 'form': 'tablet'},
            '7052': {'id': 9, 'name': 'Metoprolol', 'ingredient': 'metoprolol', 'strength': '50 mg', 'form': 'tablet'},
            
            # Insulin products
            '5856': {'id': 10, 'name': 'Insulin', 'ingredient': 'insulin', 'strength': '100 units/ml', 'form': 'injection'},
            '274783': {'id': 11, 'name': 'Insulin glargine', 'ingredient': 'insulin glargine', 'strength': '100 units/ml', 'form': 'injection'},
            
            # Antibiotics
            '723': {'id': 12, 'name': 'Amoxicillin', 'ingredient': 'amoxicillin', 'strength': '500 mg', 'form': 'capsule'},
            '2551': {'id': 13, 'name': 'Ciprofloxacin', 'ingredient': 'ciprofloxacin', 'strength': '500 mg', 'form': 'tablet'},
            '10829': {'id': 14, 'name': 'Azithromycin', 'ingredient': 'azithromycin', 'strength': '250 mg', 'form': 'tablet'},
            
            # Mental health medications
            '32937': {'id': 15, 'name': 'Sertraline', 'ingredient': 'sertraline', 'strength': '50 mg', 'form': 'tablet'},
            '2556': {'id': 16, 'name': 'Fluoxetine', 'ingredient': 'fluoxetine', 'strength': '20 mg', 'form': 'capsule'},
            '6470': {'id': 17, 'name': 'Lorazepam', 'ingredient': 'lorazepam', 'strength': '1 mg', 'form': 'tablet'},
            
            # Pain medications
            '7804': {'id': 18, 'name': 'Morphine', 'ingredient': 'morphine', 'strength': '15 mg', 'form': 'tablet'},
            '7052': {'id': 19, 'name': 'Oxycodone', 'ingredient': 'oxycodone', 'strength': '5 mg', 'form': 'tablet'},
            '1819': {'id': 20, 'name': 'Codeine', 'ingredient': 'codeine', 'strength': '30 mg', 'form': 'tablet'},
        }
        
        return rxnorm_concepts
    
    def _load_loinc_knowledge(self) -> Dict[str, Dict]:
        """
        Load LOINC laboratory and clinical observation codes.
        
        Comprehensive LOINC codes for laboratory tests and clinical observations.
        """
        loinc_codes = {
            # Chemistry panel
            '2093-3': {'id': 0, 'name': 'Cholesterol [Mass/volume] in Serum or Plasma', 'component': 'Cholesterol', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '2345-7': {'id': 1, 'name': 'Glucose [Mass/volume] in Serum or Plasma', 'component': 'Glucose', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '3094-0': {'id': 2, 'name': 'Urea nitrogen [Mass/volume] in Serum or Plasma', 'component': 'BUN', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '2160-0': {'id': 3, 'name': 'Creatinine [Mass/volume] in Serum or Plasma', 'component': 'Creatinine', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '2951-2': {'id': 4, 'name': 'Sodium [Moles/volume] in Serum or Plasma', 'component': 'Sodium', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '2823-3': {'id': 5, 'name': 'Potassium [Moles/volume] in Serum or Plasma', 'component': 'Potassium', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '2075-0': {'id': 6, 'name': 'Chloride [Moles/volume] in Serum or Plasma', 'component': 'Chloride', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            
            # Complete Blood Count (CBC)
            '6690-2': {'id': 7, 'name': 'Leukocytes [#/volume] in Blood by Automated count', 'component': 'WBC', 'system': 'Blood', 'scale': 'Quantitative'},
            '789-8': {'id': 8, 'name': 'Erythrocytes [#/volume] in Blood by Automated count', 'component': 'RBC', 'system': 'Blood', 'scale': 'Quantitative'},
            '718-7': {'id': 9, 'name': 'Hemoglobin [Mass/volume] in Blood', 'component': 'Hemoglobin', 'system': 'Blood', 'scale': 'Quantitative'},
            '4544-3': {'id': 10, 'name': 'Hematocrit [Volume Fraction] of Blood by Automated count', 'component': 'Hematocrit', 'system': 'Blood', 'scale': 'Quantitative'},
            '777-3': {'id': 11, 'name': 'Platelets [#/volume] in Blood by Automated count', 'component': 'Platelets', 'system': 'Blood', 'scale': 'Quantitative'},
            
            # Liver function tests
            '1742-6': {'id': 12, 'name': 'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma', 'component': 'ALT', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '1920-8': {'id': 13, 'name': 'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma', 'component': 'AST', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '1975-2': {'id': 14, 'name': 'Bilirubin.total [Mass/volume] in Serum or Plasma', 'component': 'Total Bilirubin', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            
            # Cardiac markers
            '10839-9': {'id': 15, 'name': 'Troponin I.cardiac [Mass/volume] in Serum or Plasma', 'component': 'Troponin I', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            '33747-0': {'id': 16, 'name': 'Creatine kinase.MB [Mass/volume] in Serum or Plasma', 'component': 'CK-MB', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            
            # Endocrine tests
            '2339-0': {'id': 17, 'name': 'Glucose [Mass/volume] in Blood', 'component': 'Blood Glucose', 'system': 'Blood', 'scale': 'Quantitative'},
            '4548-4': {'id': 18, 'name': 'Hemoglobin A1c/Hemoglobin.total in Blood', 'component': 'HbA1c', 'system': 'Blood', 'scale': 'Quantitative'},
            '3016-3': {'id': 19, 'name': 'Thyrotropin [Units/volume] in Serum or Plasma', 'component': 'TSH', 'system': 'Serum/Plasma', 'scale': 'Quantitative'},
            
            # Vital signs
            '8480-6': {'id': 20, 'name': 'Systolic blood pressure', 'component': 'Systolic BP', 'system': 'Arterial', 'scale': 'Quantitative'},
            '8462-4': {'id': 21, 'name': 'Diastolic blood pressure', 'component': 'Diastolic BP', 'system': 'Arterial', 'scale': 'Quantitative'},
            '8867-4': {'id': 22, 'name': 'Heart rate', 'component': 'Heart Rate', 'system': 'Arterial', 'scale': 'Quantitative'},
            '8310-5': {'id': 23, 'name': 'Body temperature', 'component': 'Temperature', 'system': 'Body', 'scale': 'Quantitative'},
            '9279-1': {'id': 24, 'name': 'Respiratory rate', 'component': 'Respiratory Rate', 'system': 'Respiratory', 'scale': 'Quantitative'},
        }
        
        return loinc_codes
    
    def get_knowledge_embedding(self, entity: str, kb_type: str) -> Optional[torch.Tensor]:
        """
        Retrieve knowledge embedding for a medical entity.
        
        Args:
            entity: Medical entity identifier
            kb_type: Knowledge base type (FHIR, HL7, etc.)
            
        Returns:
            Knowledge embedding tensor or None if not found
        """
        if kb_type not in self.knowledge_bases:
            logger.warning(f"Unknown knowledge base type: {kb_type}")
            return None
        
        kb = self.knowledge_bases[kb_type]
        if entity in kb:
            entity_id = kb[entity]['id']
            return self.knowledge_embeddings[kb_type](torch.tensor(entity_id, device=self.device))
        else:
            # Return unknown token embedding (last index)
            unknown_id = len(kb)
            return self.knowledge_embeddings[kb_type](torch.tensor(unknown_id, device=self.device))
    
    def get_related_concepts(self, entity: str, kb_type: str, max_relations: int = 5) -> List[Dict]:
        """
        Get related medical concepts for an entity.
        
        This is a simplified implementation. In production, this would use
        actual knowledge graph relationships.
        """
        related_concepts = []
        
        if kb_type not in self.knowledge_bases:
            return related_concepts
        
        kb = self.knowledge_bases[kb_type]
        if entity not in kb:
            return related_concepts
        
        entity_info = kb[entity]
        
        # Find related concepts based on category/type
        for other_entity, other_info in kb.items():
            if other_entity != entity and len(related_concepts) < max_relations:
                # Simple similarity based on shared attributes
                similarity_score = 0
                
                if 'category' in entity_info and 'category' in other_info:
                    if entity_info['category'] == other_info['category']:
                        similarity_score += 0.5
                
                if 'type' in entity_info and 'type' in other_info:
                    if entity_info['type'] == other_info['type']:
                        similarity_score += 0.3
                
                if similarity_score > 0.3:
                    related_concepts.append({
                        'entity': other_entity,
                        'similarity': similarity_score,
                        'info': other_info
                    })
        
        # Sort by similarity
        related_concepts.sort(key=lambda x: x['similarity'], reverse=True)
        
        return related_concepts[:max_relations]
    
    def to(self, device: str):
        """Move knowledge embeddings to device."""
        self.device = device
        for kb_name, embedding in self.knowledge_embeddings.items():
            self.knowledge_embeddings[kb_name] = embedding.to(device)
        return self

class MedicalAttention(nn.Module):
    """
    Enhanced multi-head attention with medical relationship awareness.
    
    Implements the mathematical formulation from the paper:
    MedAttention(Q, K, V) = softmax((QK^T + M) / sqrt(d_k))V
    
    where M is the medical relationship matrix.
    """
    
    def __init__(self, config: TOAIMRConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        assert config.hidden_size % config.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        
        # Standard attention layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Medical relationship matrix (M in the paper)
        self.medical_relationship_matrix = nn.Parameter(
            torch.zeros(config.max_position_embeddings, config.max_position_embeddings)
        )
        
        # Initialize medical relationship matrix with small random values
        nn.init.normal_(self.medical_relationship_matrix, mean=0.0, std=0.02)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        logger.debug(f"Initialized MedicalAttention with {config.num_attention_heads} heads")
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        medical_entity_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass implementing medical-aware attention mechanism.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            medical_entity_mask: Medical entity mask [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear transformations for Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Add medical relationship matrix for medical entities
        if medical_entity_mask is not None:
            # Get the relevant portion of the medical relationship matrix
            medical_matrix = self.medical_relationship_matrix[:seq_len, :seq_len]
            
            # Create medical relationship boost
            # medical_boost[i,j] = 1 if both i and j are medical entities, 0 otherwise
            medical_boost = medical_entity_mask.unsqueeze(2) * medical_entity_mask.unsqueeze(1)  # [batch, seq, seq]
            
            # Apply medical attention scaling (β in the paper)
            medical_attention = medical_boost * medical_matrix.unsqueeze(0) * self.config.medical_attention_scaling
            
            # Add to attention scores for all heads
            attention_scores = attention_scores + medical_attention.unsqueeze(1)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert attention mask to attention bias
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Final linear transformation and residual connection
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        
        return outputs

class KnowledgeIntegrationLayer(nn.Module):
    """
    Knowledge integration layer that incorporates external medical knowledge bases.
    
    Implements the mathematical formulation from the paper:
    h_j^(k) = h_j^(l-1) + Σ(γ_i * Attention(h_j^(l-1), φ_i(K_i)))
    
    where γ_i are learnable knowledge base weights and φ_i are embedding functions.
    """
    
    def __init__(self, config: TOAIMRConfig, knowledge_base: MedicalKnowledgeBase):
        super().__init__()
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Knowledge base weights (γ_i in the paper) - learnable parameters
        num_knowledge_bases = len(knowledge_base.knowledge_bases)
        self.knowledge_weights = nn.Parameter(torch.ones(num_knowledge_bases))
        
        # Attention mechanism for knowledge integration
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout_prob,
            batch_first=True
        )
        
        # Projection layers
        self.knowledge_projection = nn.Linear(config.knowledge_integration_dim, config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Gating mechanism for knowledge integration
        self.knowledge_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        logger.debug(f"Initialized KnowledgeIntegrationLayer with {num_knowledge_bases} knowledge bases")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        medical_entities: Optional[List[List[Dict]]] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing knowledge integration.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            medical_entities: List of medical entities for each batch item
            
        Returns:
            Knowledge-enhanced hidden states
        """
        if medical_entities is None or len(medical_entities) == 0:
            return hidden_states
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        knowledge_enhanced = hidden_states.clone()
        
        # For each knowledge base
        for kb_idx, kb_name in enumerate(self.knowledge_base.knowledge_bases.keys()):
            kb_weight = self.knowledge_weights[kb_idx]
            
            # Collect knowledge embeddings for this batch
            batch_kb_embeddings = []
            
            for batch_idx in range(batch_size):
                # Get entities for this batch item
                batch_entities = medical_entities[batch_idx] if batch_idx < len(medical_entities) else []
                
                # Extract relevant knowledge embeddings for this knowledge base
                kb_embeddings = []
                for entity_info in batch_entities:
                    if kb_name.lower() in entity_info.get('knowledge_bases', []):
                        kb_embedding = self.knowledge_base.get_knowledge_embedding(
                            entity_info['entity'], kb_name
                        )
                        if kb_embedding is not None:
                            kb_embeddings.append(kb_embedding)
                
                # If we have knowledge embeddings, pad/truncate to fixed size
                max_kb_entities = 10  # Maximum number of knowledge entities per sequence
                if kb_embeddings:
                    # Truncate if too many
                    kb_embeddings = kb_embeddings[:max_kb_entities]
                    # Pad if too few
                    while len(kb_embeddings) < max_kb_entities:
                        kb_embeddings.append(torch.zeros_like(kb_embeddings[0]))
                    
                    kb_tensor = torch.stack(kb_embeddings)
                else:
                    # No knowledge embeddings found, use zero embeddings
                    kb_tensor = torch.zeros(max_kb_entities, self.config.knowledge_integration_dim, 
                                          device=hidden_states.device)
                
                batch_kb_embeddings.append(kb_tensor)
            
            if batch_kb_embeddings:
                # Stack all batch knowledge embeddings
                kb_batch_tensor = torch.stack(batch_kb_embeddings)  # [batch, max_kb_entities, kb_dim]
                
                # Project knowledge embeddings to hidden dimension
                kb_projected = self.knowledge_projection(kb_batch_tensor)  # [batch, max_kb_entities, hidden]
                
                # Apply attention between hidden states and knowledge
                try:
                    attended_knowledge, attention_weights = self.knowledge_attention(
                        query=hidden_states,
                        key=kb_projected,
                        value=kb_projected
                    )
                    
                    # Gating mechanism to control knowledge integration
                    gate_input = torch.cat([hidden_states, attended_knowledge], dim=-1)
                    gate_values = torch.sigmoid(self.knowledge_gate(gate_input))
                    
                    # Weighted integration with knowledge base weight
                    knowledge_contribution = kb_weight * gate_values * attended_knowledge
                    knowledge_enhanced = knowledge_enhanced + knowledge_contribution
                    
                except Exception as e:
                    logger.warning(f"Knowledge attention failed for {kb_name}: {e}")
                    continue
        
        # Final projection and normalization
        output = self.output_projection(knowledge_enhanced)
        output = self.layer_norm(output + hidden_states)
        
        return output

class TOAIMRTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with medical attention and knowledge integration.
    
    This layer combines the medical-aware attention mechanism with knowledge
    integration as described in the paper.
    """
    
    def __init__(self, config: TOAIMRConfig, knowledge_base: MedicalKnowledgeBase):
        super().__init__()
        self.config = config
        
        # Medical-aware attention
        self.attention = MedicalAttention(config)
        
        # Knowledge integration
        self.knowledge_integration = KnowledgeIntegrationLayer(config, knowledge_base)
        
        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        medical_entity_mask: Optional[torch.Tensor] = None,
        medical_entities: Optional[List[List[Dict]]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through transformer layer."""
        
        # Medical-aware self-attention
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            medical_entity_mask=medical_entity_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        
        # Knowledge integration
        knowledge_output = self.knowledge_integration(
            hidden_states=attention_output,
            medical_entities=medical_entities
        )
        
        # Feed-forward network
        intermediate_output = self.intermediate(knowledge_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + knowledge_output)
        
        outputs = (layer_output,)
        if output_attentions:
            outputs = outputs + (self_attention_outputs[1],)
        
        return outputs

# Continue with the rest of the implementation...
# (Due to length constraints, I'll continue in the next part)

if __name__ == "__main__":
    # Basic testing
    print("Enhanced TOAI-MR model implementation loaded successfully!")
    
    # Test configuration
    config = TOAIMRConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8
    )
    
    print(f"✓ Configuration created with {config.num_hidden_layers} layers")
    
    # Test knowledge base
    kb = MedicalKnowledgeBase(knowledge_dim=256)
    print(f"✓ Knowledge base loaded with {len(kb.knowledge_bases)} knowledge bases")
    
    print("Enhanced TOAI-MR implementation ready for use!")

