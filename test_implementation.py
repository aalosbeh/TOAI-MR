#!/usr/bin/env python3

import sys
import os
import torch
import json
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import MedicalDataGenerator
from toai_mr_model import TOAIMRModel, TOAI_MR_CONFIG
from training_pipeline import EHRTranslationDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_generation():
    """Test synthetic dataset generation."""
    logger.info("Testing dataset generation...")
    
    try:
        generator = MedicalDataGenerator(seed=42)
        
        # Generate a small test dataset
        test_data = []
        for i in range(5):
            sample = generator.generate_training_pair()
            test_data.append(sample)
        
        # Save test data
        os.makedirs('data', exist_ok=True)
        with open('data/test_sample.json', 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        logger.info(f" Dataset generation successful! Generated {len(test_data)} samples")
        
        # Print sample
        sample = test_data[0]
        logger.info(f"Sample source format: {sample['source_format']}")
        logger.info(f"Sample target format: {sample['target_format']}")
        
        return True
        
    except Exception as e:
        logger.error(f" Dataset generation failed: {str(e)}")
        return False

def test_model_initialization():
    """Test TOAI-MR model initialization."""
    logger.info("Testing model initialization...")
    
    try:
        # Initialize model
        model = TOAIMRModel(TOAI_MR_CONFIG)
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f" Model initialization successful!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: ~{total_params * 4 / 1024**3:.2f} GB (FP32)")
        
        return model
        
    except Exception as e:
        logger.error(f" Model initialization failed: {str(e)}")
        return None

def test_forward_pass(model):
    """Test model forward pass."""
    logger.info("Testing model forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        seq_length = 64  # Reduced for testing
        
        source_ids = torch.randint(0, 1000, (batch_size, seq_length))
        target_ids = torch.randint(0, 1000, (batch_size, seq_length))
        source_mask = torch.ones(batch_size, seq_length)
        target_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(source_ids, target_ids, source_mask, target_mask)
        
        logger.info(f" Forward pass successful!")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        logger.info(f"Confidence scores shape: {outputs['confidence_scores'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f" Forward pass failed: {str(e)}")
        return False

def test_generation(model):
    """Test text generation."""
    logger.info("Testing text generation...")
    
    try:
        # Create dummy input
        batch_size = 1
        seq_length = 32
        
        source_ids = torch.randint(0, 1000, (batch_size, seq_length))
        source_mask = torch.ones(batch_size, seq_length)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                source_ids,
                source_mask,
                max_length=64
            )
        
        logger.info(f" Text generation successful!")
        logger.info(f"Generated text length: {len(generated['generated_text'])}")
        logger.info(f"Average confidence: {generated['confidence_scores'].mean():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f" Text generation failed: {str(e)}")
        return False

def test_dataset_loading():
    """Test dataset loading."""
    logger.info("Testing dataset loading...")
    
    try:
        # Check if test data exists
        if not os.path.exists('data/test_sample.json'):
            logger.warning("Test data not found, skipping dataset loading test")
            return True
        
        # Initialize model for tokenizer
        model = TOAIMRModel(TOAI_MR_CONFIG)
        
        # Create dataset
        dataset = EHRTranslationDataset('data/test_sample.json', model.tokenizer, max_length=128)
        
        # Test data loading
        sample = dataset[0]
        
        logger.info(f" Dataset loading successful!")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Source input shape: {sample['source_input_ids'].shape}")
        logger.info(f"Target input shape: {sample['target_input_ids'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f" Dataset loading failed: {str(e)}")
        return False

def test_tokenization():
    """Test medical tokenization."""
    logger.info("Testing medical tokenization...")
    
    try:
        model = TOAIMRModel(TOAI_MR_CONFIG)
        tokenizer = model.tokenizer
        
        # Test medical text
        medical_text = "PATIENT: John Doe | MRN: 12345 | DIAGNOSIS: I10 Essential hypertension | MEDICATION: Lisinopril 10mg once daily"
        
        # Tokenize
        encoded = tokenizer.encode(medical_text, max_length=128)
        decoded = tokenizer.decode(encoded['input_ids'].squeeze())
        
        logger.info(f" Tokenization successful!")
        logger.info(f"Original text length: {len(medical_text)}")
        logger.info(f"Encoded tokens: {encoded['input_ids'].shape}")
        logger.info(f"Decoded text length: {len(decoded)}")
        
        return True
        
    except Exception as e:
        logger.error(f" Tokenization failed: {str(e)}")
        return False

def generate_demo_data():
    """Generate demonstration data."""
    logger.info("Generating demonstration data...")
    
    try:
        generator = MedicalDataGenerator(seed=123)
        
        # Generate demo samples
        demo_samples = []
        for i in range(3):
            sample = generator.generate_training_pair()
            demo_samples.append(sample)
        
        # Save demo data
        os.makedirs('data', exist_ok=True)
        with open('data/demo_samples.json', 'w') as f:
            json.dump(demo_samples, f, indent=2, default=str)
        
        logger.info(f" Demo data generated! Saved {len(demo_samples)} samples")
        
        # Print one example
        sample = demo_samples[0]
        logger.info(f"\n📋 Demo Sample:")
        logger.info(f"Source Format: {sample['source_format']}")
        logger.info(f"Target Format: {sample['target_format']}")
        logger.info(f"Patient ID: {sample['metadata']['patient_id']}")
        
        return True
        
    except Exception as e:
        logger.error(f" Demo data generation failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Starting TOAI-MR Implementation Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test 1: Dataset Generation
    test_results['dataset_generation'] = test_dataset_generation()
    
    # Test 2: Model Initialization
    model = test_model_initialization()
    test_results['model_initialization'] = model is not None
    
    if model:
        # Test 3: Forward Pass
        test_results['forward_pass'] = test_forward_pass(model)
        
        # Test 4: Text Generation
        test_results['generation'] = test_generation(model)
        
        # Test 5: Tokenization
        test_results['tokenization'] = test_tokenization()
    
    # Test 6: Dataset Loading
    test_results['dataset_loading'] = test_dataset_loading()
    
    # Test 7: Demo Data Generation
    test_results['demo_data'] = generate_demo_data()
    
    # Summary
    logger.info("=" * 50)
    logger.info("🏁 Test Summary:")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = " PASS" if result else " FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! TOAI-MR implementation is working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Please check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Set device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

