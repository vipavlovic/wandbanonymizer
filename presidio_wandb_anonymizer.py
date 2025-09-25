#!/usr/bin/env python3
"""
W&B Log Anonymizer with Microsoft Presidio PII Detection

This enhanced version uses Microsoft Presidio to automatically detect and anonymize
PII in W&B logs, providing more comprehensive and accurate anonymization than
rule-based approaches.
"""

import json
import re
import hashlib
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import uuid

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("⚠️  Presidio not installed. Run: pip install presidio-analyzer presidio-anonymizer")
    print("   Falling back to rule-based anonymization...")

class PresidioWandBAnonymizer:
    def __init__(self, seed: str = "presidio_wandb_2024"):
        """
        Initialize the Presidio-powered W&B anonymizer.
        
        Args:
            seed: Seed for consistent anonymization
        """
        self.seed = seed
        self.consistent_mapping = {}
        
        if PRESIDIO_AVAILABLE:
            self.setup_presidio()
        else:
            # Fallback to basic rule-based approach
            self.setup_fallback()
    
    def setup_presidio(self):
        """Set up Presidio analyzer and anonymizer with W&B-specific patterns."""
        
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Add custom recognizers for W&B-specific patterns
        self.add_wandb_recognizers()
        
        # Configure entity types to detect
        self.entity_types = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
            "IP_ADDRESS", "LOCATION", "DATE_TIME", "URL", "DOMAIN_NAME",
            "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_ITIN",
            "IBAN_CODE", "CRYPTO", "MEDICAL_LICENSE", "UK_NHS",
            # Custom W&B entities
            "WANDB_RUN_ID", "WANDB_PROJECT", "FILE_PATH", "GIT_REPO",
            "HOSTNAME", "USERNAME", "API_KEY"
        ]
    
    def add_wandb_recognizers(self):
        """Add custom recognizers for W&B-specific PII patterns."""
        
        # W&B Run ID recognizer
        run_id_pattern = Pattern(
            name="wandb_run_id",
            regex=r'\b[a-z0-9]{8}\b',  # W&B run IDs are typically 8-char alphanumeric
            score=0.6
        )
        wandb_run_recognizer = PatternRecognizer(
            supported_entity="WANDB_RUN_ID",
            patterns=[run_id_pattern]
        )
        
        # File path recognizer (Unix and Windows)
        file_path_pattern = Pattern(
            name="file_path",
            regex=r'(?:/[^/\s]+)+/?|[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            score=0.7
        )
        file_path_recognizer = PatternRecognizer(
            supported_entity="FILE_PATH",
            patterns=[file_path_pattern]
        )
        
        # Git repository recognizer
        git_repo_pattern = Pattern(
            name="git_repo",
            regex=r'(?:git@|https?://)[\w\.-]+[:/][\w\.-]+/[\w\.-]+(?:\.git)?',
            score=0.8
        )
        git_repo_recognizer = PatternRecognizer(
            supported_entity="GIT_REPO", 
            patterns=[git_repo_pattern]
        )
        
        # Hostname recognizer
        hostname_pattern = Pattern(
            name="hostname",
            regex=r'\b[\w-]+\.[\w.-]+\.[a-z]{2,}\b',
            score=0.6
        )
        hostname_recognizer = PatternRecognizer(
            supported_entity="HOSTNAME",
            patterns=[hostname_pattern]
        )
        
        # Username recognizer (common username patterns)
        username_pattern = Pattern(
            name="username",
            regex=r'\b[a-zA-Z][a-zA-Z0-9._-]{2,20}\b',
            score=0.4  # Lower score to avoid false positives
        )
        username_recognizer = PatternRecognizer(
            supported_entity="USERNAME",
            patterns=[username_pattern]
        )
        
        # API Key recognizer (various formats)
        api_key_pattern = Pattern(
            name="api_key",
            regex=r'\b(?:sk-|pk-|rk-)?[A-Za-z0-9]{20,64}\b',
            score=0.7
        )
        api_key_recognizer = PatternRecognizer(
            supported_entity="API_KEY",
            patterns=[api_key_pattern]
        )
        
        # Add all custom recognizers
        self.analyzer.registry.add_recognizer(wandb_run_recognizer)
        self.analyzer.registry.add_recognizer(file_path_recognizer)
        self.analyzer.registry.add_recognizer(git_repo_recognizer)
        self.analyzer.registry.add_recognizer(hostname_recognizer)
        self.analyzer.registry.add_recognizer(username_recognizer)
        self.analyzer.registry.add_recognizer(api_key_recognizer)
    
    def setup_fallback(self):
        """Set up fallback rule-based anonymization when Presidio is not available."""
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.path_pattern = re.compile(r'(/[^/\s]+)+/?')
        self.windows_path_pattern = re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*')
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    
    def _generate_consistent_replacement(self, text: str, entity_type: str) -> str:
        """Generate a consistent replacement for detected PII."""
        cache_key = f"{entity_type}:{text}"
        
        if cache_key in self.consistent_mapping:
            return self.consistent_mapping[cache_key]
        
        # Generate hash-based consistent replacement
        hash_obj = hashlib.md5(f"{self.seed}:{cache_key}".encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        # Create replacement based on entity type
        if entity_type == "PERSON":
            replacement = f"Person_{hash_hex}"
        elif entity_type == "EMAIL_ADDRESS":
            replacement = f"user_{hash_hex}@example.com"
        elif entity_type == "PHONE_NUMBER":
            replacement = f"555-{hash_hex[:3]}-{hash_hex[3:7]}"
        elif entity_type == "URL" or entity_type == "GIT_REPO":
            replacement = f"https://example.com/repo_{hash_hex}"
        elif entity_type == "IP_ADDRESS":
            # Generate fake IP
            nums = [int(hash_hex[i:i+2], 16) % 255 for i in range(0, 8, 2)]
            replacement = f"192.168.{nums[0]}.{nums[1]}"
        elif entity_type == "FILE_PATH":
            replacement = f"/anonymized/path_{hash_hex}"
        elif entity_type == "HOSTNAME":
            replacement = f"host-{hash_hex}.example.com"
        elif entity_type == "USERNAME":
            replacement = f"user_{hash_hex}"
        elif entity_type == "WANDB_RUN_ID":
            # Generate new W&B-style run ID
            new_hash = hashlib.md5(f"{self.seed}:run:{text}".encode()).hexdigest()[:8]
            replacement = new_hash
        elif entity_type == "API_KEY":
            replacement = f"sk-{hash_hex}{'x' * (len(text) - len(hash_hex) - 3)}"
        elif entity_type == "LOCATION":
            replacement = f"City_{hash_hex}"
        elif entity_type == "DATE_TIME":
            replacement = "2024-01-01T00:00:00Z"  # Generic timestamp
        else:
            # Generic replacement
            replacement = f"anon_{entity_type.lower()}_{hash_hex}"
        
        self.consistent_mapping[cache_key] = replacement
        return replacement
    
    def analyze_and_anonymize_text(self, text: str) -> str:
        """Analyze text with Presidio and anonymize detected PII."""
        if not PRESIDIO_AVAILABLE:
            return self.fallback_anonymize_text(text)
        
        if not isinstance(text, str) or not text.strip():
            return text
        
        try:
            # Analyze text for PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=self.entity_types,
                language='en'
            )
            
            if not analyzer_results:
                return text
            
            # Create anonymization operators with consistent replacements
            operators = {}
            for result in analyzer_results:
                entity_text = text[result.start:result.end]
                consistent_replacement = self._generate_consistent_replacement(
                    entity_text, result.entity_type
                )
                operators[result.entity_type] = OperatorConfig(
                    "replace", {"new_value": consistent_replacement}
                )
            
            # Anonymize the text
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            
            return anonymized_result.text
            
        except Exception as e:
            print(f"⚠️  Presidio error: {e}. Falling back to rule-based anonymization.")
            return self.fallback_anonymize_text(text)
    
    def fallback_anonymize_text(self, text: str) -> str:
        """Fallback rule-based anonymization when Presidio fails."""
        if not isinstance(text, str):
            return text
        
        # Replace emails
        text = self.email_pattern.sub(
            lambda m: self._generate_consistent_replacement(m.group(), "EMAIL_ADDRESS"), 
            text
        )
        
        # Replace URLs
        text = self.url_pattern.sub(
            lambda m: self._generate_consistent_replacement(m.group(), "URL"),
            text
        )
        
        # Replace file paths
        if not text.startswith(('http://', 'https://')):
            text = self.path_pattern.sub(
                lambda m: self._generate_consistent_replacement(m.group(), "FILE_PATH"),
                text
            )
            text = self.windows_path_pattern.sub(
                lambda m: self._generate_consistent_replacement(m.group(), "FILE_PATH"),
                text
            )
        
        return text
    
    def anonymize_json_object(self, obj: Any) -> Any:
        """Recursively anonymize a JSON object using Presidio."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Anonymize both keys and values that might contain PII
                anonymized_key = key
                if isinstance(key, str):
                    anonymized_key = self.analyze_and_anonymize_text(key)
                
                anonymized_value = self.anonymize_json_object(value)
                result[anonymized_key] = anonymized_value
            return result
        
        elif isinstance(obj, list):
            return [self.anonymize_json_object(item) for item in obj]
        
        elif isinstance(obj, str):
            return self.analyze_and_anonymize_text(obj)
        
        else:
            return obj
    
    def anonymize_file(self, input_path: str, output_path: str = None):
        """Anonymize a W&B log file using Presidio."""
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"anon_{input_path.name}"
        else:
            output_path = Path(output_path)
        
        print(f"🔍 Analyzing and anonymizing {input_path} -> {output_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.suffix.lower() == '.json':
                    # Handle JSON files
                    data = json.load(f)
                    anonymized_data = self.anonymize_json_object(data)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(anonymized_data, out_f, indent=2, ensure_ascii=False)
                
                elif input_path.suffix.lower() == '.jsonl':
                    # Handle JSONL files
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                anonymized_data = self.anonymize_json_object(data)
                                out_f.write(json.dumps(anonymized_data, ensure_ascii=False) + '\n')
                
                else:
                    # Handle text files
                    content = f.read()
                    anonymized_content = self.analyze_and_anonymize_text(content)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(anonymized_content)
        
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
            raise
    
    def anonymize_directory(self, input_dir: str, output_dir: str = None):
        """Anonymize all supported files in a directory using Presidio."""
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir.parent / f"anon_{input_dir.name}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions commonly found in W&B logs
        supported_extensions = {'.json', '.jsonl', '.txt', '.log', '.yaml', '.yml'}
        
        print(f"🔍 Scanning directory: {input_dir}")
        files_processed = 0
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Maintain directory structure
                relative_path = file_path.relative_to(input_dir)
                output_file_path = output_dir / relative_path
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.anonymize_file(file_path, output_file_path)
                files_processed += 1
        
        print(f"✅ Processed {files_processed} files")
    
    def get_pii_statistics(self, input_path: str) -> Dict[str, int]:
        """Analyze a file/directory and return statistics on detected PII."""
        if not PRESIDIO_AVAILABLE:
            return {"error": "Presidio not available for PII statistics"}
        
        input_path = Path(input_path)
        pii_stats = {}
        
        def analyze_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix.lower() == '.json':
                        data = json.load(f)
                        text = json.dumps(data, ensure_ascii=False)
                    elif file_path.suffix.lower() == '.jsonl':
                        lines = f.readlines()
                        text = '\n'.join(lines)
                    else:
                        text = f.read()
                
                # Analyze with Presidio
                results = self.analyzer.analyze(
                    text=text,
                    entities=self.entity_types,
                    language='en'
                )
                
                for result in results:
                    entity_type = result.entity_type
                    pii_stats[entity_type] = pii_stats.get(entity_type, 0) + 1
                    
            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")
        
        if input_path.is_file():
            analyze_file(input_path)
        else:
            supported_extensions = {'.json', '.jsonl', '.txt', '.log', '.yaml', '.yml'}
            for file_path in input_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    analyze_file(file_path)
        
        return pii_stats
    
    def save_mapping(self, output_path: str):
        """Save the anonymization mapping to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.consistent_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"🗂️  Anonymization mapping saved to {output_path}")
        print("⚠️  WARNING: Keep this mapping file secure - it can reverse anonymization!")

def main():
    parser = argparse.ArgumentParser(
        description="Anonymize W&B logs using Microsoft Presidio PII detection"
    )
    parser.add_argument("input", help="Input file or directory to anonymize")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("-s", "--seed", default="presidio_wandb_2024", 
                       help="Seed for consistent anonymization")
    parser.add_argument("-m", "--save-mapping", 
                       help="Save anonymization mapping to specified file")
    parser.add_argument("--stats", action="store_true",
                       help="Show PII detection statistics before anonymizing")
    
    args = parser.parse_args()
    
    if not PRESIDIO_AVAILABLE:
        print("⚠️  Presidio not available. Install with:")
        print("   pip install presidio-analyzer presidio-anonymizer")
        print("   Continuing with rule-based anonymization...")
    
    anonymizer = PresidioWandBAnonymizer(seed=args.seed)
    
    input_path = Path(args.input)
    
    # Show PII statistics if requested
    if args.stats and PRESIDIO_AVAILABLE:
        print("🔍 Analyzing PII in input files...")
        stats = anonymizer.get_pii_statistics(args.input)
        print("📊 PII Detection Statistics:")
        for entity_type, count in sorted(stats.items()):
            print(f"   {entity_type}: {count} instances")
        print()
    
    # Perform anonymization
    if input_path.is_file():
        anonymizer.anonymize_file(args.input, args.output)
    elif input_path.is_dir():
        anonymizer.anonymize_directory(args.input, args.output)
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")
        return 1
    
    if args.save_mapping:
        anonymizer.save_mapping(args.save_mapping)
    
    print("✅ Anonymization complete!")
    return 0

if __name__ == "__main__":
    exit(main())
