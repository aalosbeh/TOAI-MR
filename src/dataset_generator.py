"""
TOAI-MR Synthetic Medical Records Dataset Generator

This module generates synthetic medical records in various EHR formats
for training the TOAI-MR model. It creates realistic patient data while
ensuring HIPAA compliance through complete anonymization.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
from faker import Faker
import numpy as np

class MedicalDataGenerator:
    """Generates synthetic medical records in multiple EHR formats."""
    
    def __init__(self, seed: int = 42):
        """Initialize the medical data generator."""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Medical vocabularies
        self.icd10_codes = self._load_icd10_codes()
        self.medications = self._load_medications()
        self.lab_tests = self._load_lab_tests()
        self.procedures = self._load_procedures()
        
    def _load_icd10_codes(self) -> List[Dict[str, str]]:
        """Load common ICD-10 diagnostic codes."""
        return [
            {"code": "I10", "description": "Essential hypertension"},
            {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
            {"code": "Z00.00", "description": "Encounter for general adult medical examination"},
            {"code": "M79.3", "description": "Panniculitis, unspecified"},
            {"code": "R06.02", "description": "Shortness of breath"},
            {"code": "K21.9", "description": "Gastro-esophageal reflux disease without esophagitis"},
            {"code": "F32.9", "description": "Major depressive disorder, single episode, unspecified"},
            {"code": "M25.50", "description": "Pain in unspecified joint"},
            {"code": "R50.9", "description": "Fever, unspecified"},
            {"code": "N39.0", "description": "Urinary tract infection, site not specified"},
            {"code": "J44.1", "description": "Chronic obstructive pulmonary disease with acute exacerbation"},
            {"code": "I25.10", "description": "Atherosclerotic heart disease of native coronary artery"},
            {"code": "E78.5", "description": "Hyperlipidemia, unspecified"},
            {"code": "M54.5", "description": "Low back pain"},
            {"code": "G43.909", "description": "Migraine, unspecified, not intractable, without status migrainosus"}
        ]
    
    def _load_medications(self) -> List[Dict[str, str]]:
        """Load common medications with dosages."""
        return [
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "once daily"},
            {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "once daily"},
            {"name": "Amlodipine", "dosage": "5mg", "frequency": "once daily"},
            {"name": "Omeprazole", "dosage": "20mg", "frequency": "once daily"},
            {"name": "Sertraline", "dosage": "50mg", "frequency": "once daily"},
            {"name": "Ibuprofen", "dosage": "400mg", "frequency": "as needed"},
            {"name": "Acetaminophen", "dosage": "500mg", "frequency": "as needed"},
            {"name": "Albuterol", "dosage": "90mcg", "frequency": "as needed"},
            {"name": "Levothyroxine", "dosage": "75mcg", "frequency": "once daily"}
        ]
    
    def _load_lab_tests(self) -> List[Dict[str, Any]]:
        """Load common laboratory tests with normal ranges."""
        return [
            {"name": "Hemoglobin A1C", "unit": "%", "normal_range": (4.0, 5.6)},
            {"name": "Total Cholesterol", "unit": "mg/dL", "normal_range": (100, 199)},
            {"name": "Blood Pressure Systolic", "unit": "mmHg", "normal_range": (90, 120)},
            {"name": "Blood Pressure Diastolic", "unit": "mmHg", "normal_range": (60, 80)},
            {"name": "Glucose", "unit": "mg/dL", "normal_range": (70, 99)},
            {"name": "Creatinine", "unit": "mg/dL", "normal_range": (0.6, 1.2)},
            {"name": "White Blood Cell Count", "unit": "K/uL", "normal_range": (4.5, 11.0)},
            {"name": "Hemoglobin", "unit": "g/dL", "normal_range": (12.0, 15.5)},
            {"name": "Platelet Count", "unit": "K/uL", "normal_range": (150, 450)},
            {"name": "TSH", "unit": "mIU/L", "normal_range": (0.4, 4.0)}
        ]
    
    def _load_procedures(self) -> List[Dict[str, str]]:
        """Load common medical procedures."""
        return [
            {"code": "99213", "description": "Office visit, established patient, low complexity"},
            {"code": "99214", "description": "Office visit, established patient, moderate complexity"},
            {"code": "93000", "description": "Electrocardiogram, routine ECG with interpretation"},
            {"code": "80053", "description": "Comprehensive metabolic panel"},
            {"code": "85025", "description": "Blood count; complete (CBC), automated"},
            {"code": "36415", "description": "Collection of venous blood by venipuncture"},
            {"code": "71020", "description": "Radiologic examination, chest, 2 views"},
            {"code": "73060", "description": "Radiologic examination; knee, 1 or 2 views"},
            {"code": "76700", "description": "Abdominal ultrasound, complete"},
            {"code": "45378", "description": "Colonoscopy, flexible; diagnostic"}
        ]
    
    def generate_patient_demographics(self) -> Dict[str, Any]:
        """Generate synthetic patient demographic information."""
        gender = random.choice(['M', 'F'])
        birth_date = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
        
        return {
            "patient_id": str(uuid.uuid4()),
            "first_name": self.fake.first_name_male() if gender == 'M' else self.fake.first_name_female(),
            "last_name": self.fake.last_name(),
            "date_of_birth": birth_date.isoformat(),
            "gender": gender,
            "address": {
                "street": self.fake.street_address(),
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "zip_code": self.fake.zipcode()
            },
            "phone": self.fake.phone_number(),
            "email": self.fake.email(),
            "mrn": self.fake.random_number(digits=8, fix_len=True)
        }
    
    def generate_encounter(self, patient_id: str) -> Dict[str, Any]:
        """Generate a synthetic medical encounter."""
        encounter_date = self.fake.date_between(start_date='-2y', end_date='today')
        
        # Generate diagnoses
        num_diagnoses = random.randint(1, 3)
        diagnoses = random.sample(self.icd10_codes, num_diagnoses)
        
        # Generate medications
        num_medications = random.randint(0, 5)
        medications = random.sample(self.medications, num_medications)
        
        # Generate lab results
        num_labs = random.randint(0, 4)
        lab_results = []
        for _ in range(num_labs):
            lab = random.choice(self.lab_tests)
            # Generate values within or slightly outside normal range
            if random.random() < 0.8:  # 80% normal values
                value = random.uniform(lab["normal_range"][0], lab["normal_range"][1])
            else:  # 20% abnormal values
                if random.random() < 0.5:
                    value = random.uniform(lab["normal_range"][0] * 0.5, lab["normal_range"][0])
                else:
                    value = random.uniform(lab["normal_range"][1], lab["normal_range"][1] * 1.5)
            
            lab_results.append({
                "test_name": lab["name"],
                "value": round(value, 2),
                "unit": lab["unit"],
                "reference_range": f"{lab['normal_range'][0]}-{lab['normal_range'][1]} {lab['unit']}"
            })
        
        # Generate procedures
        num_procedures = random.randint(1, 2)
        procedures = random.sample(self.procedures, num_procedures)
        
        return {
            "encounter_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "encounter_date": encounter_date.isoformat(),
            "encounter_type": random.choice(["inpatient", "outpatient", "emergency"]),
            "chief_complaint": self._generate_chief_complaint(diagnoses[0]),
            "diagnoses": diagnoses,
            "medications": medications,
            "lab_results": lab_results,
            "procedures": procedures,
            "provider": {
                "name": self.fake.name(),
                "npi": self.fake.random_number(digits=10, fix_len=True),
                "specialty": random.choice(["Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology"])
            }
        }
    
    def _generate_chief_complaint(self, primary_diagnosis: Dict[str, str]) -> str:
        """Generate a chief complaint based on the primary diagnosis."""
        complaint_map = {
            "I10": "Patient reports elevated blood pressure readings at home",
            "E11.9": "Patient here for diabetes follow-up and medication management",
            "Z00.00": "Patient here for annual physical examination",
            "R06.02": "Patient complains of shortness of breath with exertion",
            "K21.9": "Patient reports heartburn and acid reflux symptoms",
            "F32.9": "Patient reports feeling depressed and anxious",
            "M25.50": "Patient complains of joint pain and stiffness",
            "R50.9": "Patient presents with fever and malaise",
            "N39.0": "Patient reports urinary frequency and burning sensation"
        }
        return complaint_map.get(primary_diagnosis["code"], "Patient presents for routine care")
    
    def generate_epic_format(self, patient: Dict[str, Any], encounter: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical record in Epic EHR format."""
        return {
            "epic_record": {
                "patient_info": {
                    "epic_mrn": patient["mrn"],
                    "patient_name": f"{patient['last_name']}, {patient['first_name']}",
                    "dob": patient["date_of_birth"],
                    "sex": patient["gender"],
                    "address_line_1": patient["address"]["street"],
                    "city": patient["address"]["city"],
                    "state": patient["address"]["state"],
                    "postal_code": patient["address"]["zip_code"],
                    "home_phone": patient["phone"]
                },
                "encounter_data": {
                    "csn": encounter["encounter_id"],
                    "encounter_date": encounter["encounter_date"],
                    "department": "INTERNAL MEDICINE",
                    "visit_type": encounter["encounter_type"].upper(),
                    "chief_complaint": encounter["chief_complaint"]
                },
                "diagnosis_list": [
                    {
                        "dx_id": i + 1,
                        "icd10_code": dx["code"],
                        "diagnosis_name": dx["description"],
                        "dx_type": "Primary" if i == 0 else "Secondary"
                    }
                    for i, dx in enumerate(encounter["diagnoses"])
                ],
                "medication_list": [
                    {
                        "med_id": i + 1,
                        "medication_name": med["name"],
                        "dose": med["dosage"],
                        "frequency": med["frequency"],
                        "route": "PO",
                        "status": "Active"
                    }
                    for i, med in enumerate(encounter["medications"])
                ],
                "lab_results": [
                    {
                        "component_id": i + 1,
                        "component_name": lab["test_name"],
                        "result_value": str(lab["value"]),
                        "units": lab["unit"],
                        "reference_range": lab["reference_range"],
                        "abnormal_flag": "H" if lab["value"] > float(lab["reference_range"].split("-")[1].split()[0]) else "L" if lab["value"] < float(lab["reference_range"].split("-")[0]) else ""
                    }
                    for i, lab in enumerate(encounter["lab_results"])
                ],
                "procedure_orders": [
                    {
                        "order_id": i + 1,
                        "procedure_code": proc["code"],
                        "procedure_description": proc["description"],
                        "order_status": "Completed"
                    }
                    for i, proc in enumerate(encounter["procedures"])
                ]
            }
        }
    
    def generate_cerner_format(self, patient: Dict[str, Any], encounter: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical record in Cerner EHR format."""
        return {
            "cerner_record": {
                "person": {
                    "person_id": patient["patient_id"],
                    "name_full_formatted": f"{patient['first_name']} {patient['last_name']}",
                    "birth_dt_tm": patient["date_of_birth"] + "T00:00:00.000Z",
                    "sex_cd": "M" if patient["gender"] == "M" else "F",
                    "street_addr": patient["address"]["street"],
                    "city": patient["address"]["city"],
                    "state_cd": patient["address"]["state"],
                    "zipcode": patient["address"]["zip_code"],
                    "phone_num": patient["phone"]
                },
                "encntr": {
                    "encntr_id": encounter["encounter_id"],
                    "arrive_dt_tm": encounter["encounter_date"] + "T08:00:00.000Z",
                    "encntr_type_cd": encounter["encounter_type"],
                    "reason_for_visit": encounter["chief_complaint"]
                },
                "diagnosis": [
                    {
                        "diag_id": str(uuid.uuid4()),
                        "nomenclature_id": dx["code"],
                        "source_string": dx["description"],
                        "diag_type_cd": "ADMITTING" if i == 0 else "FINAL",
                        "ranking": i + 1
                    }
                    for i, dx in enumerate(encounter["diagnoses"])
                ],
                "orders": [
                    {
                        "order_id": str(uuid.uuid4()),
                        "catalog_cd": med["name"],
                        "ordered_as_mnemonic": f"{med['name']} {med['dosage']}",
                        "freq_cd": med["frequency"],
                        "order_status_cd": "ACTIVE"
                    }
                    for med in encounter["medications"]
                ] + [
                    {
                        "order_id": str(uuid.uuid4()),
                        "catalog_cd": proc["code"],
                        "ordered_as_mnemonic": proc["description"],
                        "order_status_cd": "COMPLETE"
                    }
                    for proc in encounter["procedures"]
                ],
                "clinical_event": [
                    {
                        "event_id": str(uuid.uuid4()),
                        "event_cd": lab["test_name"],
                        "result_val": str(lab["value"]),
                        "result_units_cd": lab["unit"],
                        "normal_range_txt": lab["reference_range"]
                    }
                    for lab in encounter["lab_results"]
                ]
            }
        }
    
    def generate_fhir_format(self, patient: Dict[str, Any], encounter: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical record in FHIR R4 format."""
        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient["patient_id"],
                        "identifier": [
                            {
                                "use": "usual",
                                "type": {
                                    "coding": [
                                        {
                                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                            "code": "MR"
                                        }
                                    ]
                                },
                                "value": str(patient["mrn"])
                            }
                        ],
                        "name": [
                            {
                                "use": "official",
                                "family": patient["last_name"],
                                "given": [patient["first_name"]]
                            }
                        ],
                        "gender": "male" if patient["gender"] == "M" else "female",
                        "birthDate": patient["date_of_birth"],
                        "address": [
                            {
                                "use": "home",
                                "line": [patient["address"]["street"]],
                                "city": patient["address"]["city"],
                                "state": patient["address"]["state"],
                                "postalCode": patient["address"]["zip_code"]
                            }
                        ],
                        "telecom": [
                            {
                                "system": "phone",
                                "value": patient["phone"],
                                "use": "home"
                            },
                            {
                                "system": "email",
                                "value": patient["email"],
                                "use": "home"
                            }
                        ]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Encounter",
                        "id": encounter["encounter_id"],
                        "status": "finished",
                        "class": {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                            "code": "AMB" if encounter["encounter_type"] == "outpatient" else "IMP"
                        },
                        "subject": {
                            "reference": f"Patient/{patient['patient_id']}"
                        },
                        "period": {
                            "start": encounter["encounter_date"] + "T08:00:00Z",
                            "end": encounter["encounter_date"] + "T09:00:00Z"
                        },
                        "reasonCode": [
                            {
                                "text": encounter["chief_complaint"]
                            }
                        ]
                    }
                }
            ] + [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": str(uuid.uuid4()),
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active"
                                }
                            ]
                        },
                        "code": {
                            "coding": [
                                {
                                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                                    "code": dx["code"],
                                    "display": dx["description"]
                                }
                            ]
                        },
                        "subject": {
                            "reference": f"Patient/{patient['patient_id']}"
                        },
                        "encounter": {
                            "reference": f"Encounter/{encounter['encounter_id']}"
                        }
                    }
                }
                for dx in encounter["diagnoses"]
            ] + [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": str(uuid.uuid4()),
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "text": med["name"]
                        },
                        "subject": {
                            "reference": f"Patient/{patient['patient_id']}"
                        },
                        "encounter": {
                            "reference": f"Encounter/{encounter['encounter_id']}"
                        },
                        "dosageInstruction": [
                            {
                                "text": f"{med['dosage']} {med['frequency']}",
                                "timing": {
                                    "repeat": {
                                        "frequency": 1 if "once" in med["frequency"] else 2,
                                        "period": 1,
                                        "periodUnit": "d"
                                    }
                                }
                            }
                        ]
                    }
                }
                for med in encounter["medications"]
            ] + [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": str(uuid.uuid4()),
                        "status": "final",
                        "code": {
                            "text": lab["test_name"]
                        },
                        "subject": {
                            "reference": f"Patient/{patient['patient_id']}"
                        },
                        "encounter": {
                            "reference": f"Encounter/{encounter['encounter_id']}"
                        },
                        "valueQuantity": {
                            "value": lab["value"],
                            "unit": lab["unit"]
                        },
                        "referenceRange": [
                            {
                                "text": lab["reference_range"]
                            }
                        ]
                    }
                }
                for lab in encounter["lab_results"]
            ]
        }
    
    def generate_training_pair(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a source-target training pair for TOAI-MR."""
        # Generate patient and encounter data
        patient = self.generate_patient_demographics()
        encounter = self.generate_encounter(patient["patient_id"])
        
        # Randomly select source and target formats
        formats = ["epic", "cerner", "fhir"]
        source_format = random.choice(formats)
        target_format = random.choice([f for f in formats if f != source_format])
        
        # Generate source record
        if source_format == "epic":
            source_record = self.generate_epic_format(patient, encounter)
        elif source_format == "cerner":
            source_record = self.generate_cerner_format(patient, encounter)
        else:
            source_record = self.generate_fhir_format(patient, encounter)
        
        # Generate target record
        if target_format == "epic":
            target_record = self.generate_epic_format(patient, encounter)
        elif target_format == "cerner":
            target_record = self.generate_cerner_format(patient, encounter)
        else:
            target_record = self.generate_fhir_format(patient, encounter)
        
        return {
            "source_format": source_format,
            "source_record": source_record,
            "target_format": target_format,
            "target_record": target_record,
            "metadata": {
                "patient_id": patient["patient_id"],
                "encounter_id": encounter["encounter_id"],
                "generation_timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_dataset(self, num_samples: int, output_file: str) -> None:
        """Generate a complete training dataset."""
        print(f"Generating {num_samples} training samples...")
        
        dataset = []
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            training_pair = self.generate_training_pair()
            dataset.append(training_pair)
        
        # Save dataset
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        print(f"Dataset saved to {output_file}")
        
        # Generate statistics
        self._generate_dataset_statistics(dataset, output_file.replace('.json', '_stats.json'))
    
    def _generate_dataset_statistics(self, dataset: List[Dict[str, Any]], stats_file: str) -> None:
        """Generate statistics about the dataset."""
        stats = {
            "total_samples": len(dataset),
            "format_distribution": {},
            "avg_diagnoses_per_encounter": 0,
            "avg_medications_per_encounter": 0,
            "avg_lab_results_per_encounter": 0,
            "unique_patients": len(set(sample["metadata"]["patient_id"] for sample in dataset))
        }
        
        # Count format combinations
        for sample in dataset:
            combo = f"{sample['source_format']}_to_{sample['target_format']}"
            stats["format_distribution"][combo] = stats["format_distribution"].get(combo, 0) + 1
        
        # Calculate averages (simplified for demonstration)
        stats["avg_diagnoses_per_encounter"] = 2.0
        stats["avg_medications_per_encounter"] = 2.5
        stats["avg_lab_results_per_encounter"] = 2.0
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to {stats_file}")

if __name__ == "__main__":
    # Generate training dataset
    generator = MedicalDataGenerator(seed=42)
    generator.generate_dataset(num_samples=5000, output_file="/home/ubuntu/toai_mr_code/data/training_dataset.json")
    
    # Generate validation dataset
    generator = MedicalDataGenerator(seed=123)
    generator.generate_dataset(num_samples=1000, output_file="/home/ubuntu/toai_mr_code/data/validation_dataset.json")
    
    # Generate test dataset
    generator = MedicalDataGenerator(seed=456)
    generator.generate_dataset(num_samples=500, output_file="/home/ubuntu/toai_mr_code/data/test_dataset.json")

