#!/usr/bin/env python3
"""
W&B Log Anonymizer with Microsoft Presidio PII Detection

This enhanced version uses Microsoft Presidio to automatically detect and anonymize
PII in W&B logs, providing more comprehensive and accurate anonymization.
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
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("⚠️  Presidio not installed. Run: pip install presidio-analyzer presidio-anonymizer")
    print("   Falling back to rule-based anonymization...")

class PresidioWandBAnonymizer:
    def __init__(self, seed: str = "presidio_wandb_2024"):
        """Initialize the Presidio-powered W&B anonymizer."""
        self.seed = seed
        self.consistent_mapping = {}
        
        if PRESIDIO_AVAILABLE:
            self.setup_presidio()
        else:
            self.setup_fallback()
    
    def setup_presidio(self):
        """Set up Presidio analyzer and anonymizer with W&B-specific patterns."""
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.add_wandb_recognizers()
        
        self.entity_types = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
            "IP_ADDRESS", "LOCATION", "DATE_TIME", "URL", "DOMAIN_NAME",
            "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_ITIN",
            "IBAN_CODE", "CRYPTO", "MEDICAL_LICENSE", "UK_NHS",
            "WANDB_RUN_ID", "WANDB_PROJECT", "FILE_PATH", "GIT_REPO",
            "HOSTNAME", "USERNAME", "API_KEY"
        ]
    
    def add_wandb_recognizers(self):
        """Add custom recognizers for W&B-specific PII patterns."""
        run_id_pattern = Pattern(name="wandb_run_id", regex=r'\b[a-z0-9]{8}\b', score=0.6)
        wandb_run_recognizer = PatternRecognizer(supported_entity="WANDB_RUN_ID", patterns=[run_id_pattern])
        
        file_path_pattern = Pattern(
            name="file_path",
            regex=r'(?:/[^/\s]+)+/?|[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            score=0.7
        )
        file_path_recognizer = PatternRecognizer(supported_entity="FILE_PATH", patterns=[file_path_pattern])
        
        git_repo_pattern = Pattern(
            name="git_repo",
            regex=r'(?:git@|https?://)[\w\.-]+[:/][\w\.-]+/[\w\.-]+(?:\.git)?',
            score=0.8
        )
        git_repo_recognizer = PatternRecognizer(supported_entity="GIT_REPO", patterns=[git_repo_pattern])
        
        hostname_pattern = Pattern(name="hostname", regex=r'\b[\w-]+\.[\w.-]+\.[a-z]{2,}\b', score=0.6)
        hostname_recognizer = PatternRecognizer(supported_entity="HOSTNAME", patterns=[hostname_pattern])
        
        username_pattern = Pattern(name="username", regex=r'\b[a-zA-Z][a-zA-Z0-9._-]{2,20}\b', score=0.4)
        username_recognizer = PatternRecognizer(supported_entity="USERNAME", patterns=[username_pattern])
        
        api_key_pattern = Pattern(name="api_key", regex=r'\b(?:sk-|pk-|rk-)?[A-Za-z0-9]{20,64}\b', score=0.7)
        api_key_recognizer = PatternRecognizer(supported_entity="API_KEY", patterns=[api_key_pattern])
        
        self.analyzer.registry.add_recognizer(wandb_run_recognizer)
        self.analyzer.registry.add_recognizer(file_path_recognizer)
        self.analyzer.registry.add_recognizer(git_repo_recognizer)
        self.analyzer.registry.add_recognizer(hostname_recognizer)
        self.analyzer.registry.add_recognizer(username_recognizer)
        self.analyzer.registry.add_recognizer(api_key_recognizer)
    
    def setup_fallback(self):
        """Set up fallback rule-based anonymization."""
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.path_pattern = re.compile(r'(/[^/\s]+)+/?')
        self.windows_path_pattern = re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*')
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    
    def _generate_consistent_replacement(self, text: str, entity_type: str) -> str:
        """Generate a consistent replacement for detected PII."""
        cache_key = f"{entity_type}:{text}"
        
        if cache_key in self.consistent_mapping:
            return self.consistent_mapping[cache_key]
        
        hash_obj = hashlib.md5(f"{self.seed}:{cache_key}".encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        if entity_type == "PERSON":
            replacement = f"Person_{hash_hex}"
        elif entity_type == "EMAIL_ADDRESS":
            replacement = f"user_{hash_hex}@example.com"
        elif entity_type == "PHONE_NUMBER":
            replacement = f"555-{hash_hex[:3]}-{hash_hex[3:7]}"
        elif entity_type == "URL" or entity_type == "GIT_REPO":
            replacement = f"https://example.com/repo_{hash_hex}"
        elif entity_type == "IP_ADDRESS":
            nums = [int(hash_hex[i:i+2], 16) % 255 for i in range(0, 8, 2)]
            replacement = f"192.168.{nums[0]}.{nums[1]}"
        elif entity_type == "FILE_PATH":
            replacement = f"/anonymized/path_{hash_hex}"
        elif entity_type == "HOSTNAME":
            replacement = f"host-{hash_hex}.example.com"
        elif entity_type == "USERNAME":
            replacement = f"user_{hash_hex}"
        elif entity_type == "WANDB_RUN_ID":
            new_hash = hashlib.md5(f"{self.seed}:run:{text}".encode()).hexdigest()[:8]
            replacement = new_hash
        elif entity_type == "API_KEY":
            replacement = f"sk-{hash_hex}{'x' * (len(text) - len(hash_hex) - 3)}"
        elif entity_type == "LOCATION":
            replacement = f"City_{hash_hex}"
        elif entity_type == "DATE_TIME":
            replacement = "2024-01-01T00:00:00Z"
        else:
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
            analyzer_results = self.analyzer.analyze(text=text, entities=self.entity_types, language='en')
            
            if not analyzer_results:
                return text
            
            operators = {}
            for result in analyzer_results:
                entity_text = text[result.start:result.end]
                consistent_replacement = self._generate_consistent_replacement(entity_text, result.entity_type)
                operators[result.entity_type] = OperatorConfig("replace", {"new_value": consistent_replacement})
            
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operators)
            return anonymized_result.text
            
        except Exception as e:
            print(f"⚠️  Presidio error: {e}. Falling back to rule-based anonymization.")
            return self.fallback_anonymize_text(text)
    
    def fallback_anonymize_text(self, text: str) -> str:
        """Fallback rule-based anonymization."""
        if not isinstance(text, str):
            return text
        
        text = self.email_pattern.sub(
            lambda m: self._generate_consistent_replacement(m.group(), "EMAIL_ADDRESS"), text
        )
        text = self.url_pattern.sub(
            lambda m: self._generate_consistent_replacement(m.group(), "URL"), text
        )
        
        if not text.startswith(('http://', 'https://')):
            text = self.path_pattern.sub(
                lambda m: self._generate_consistent_replacement(m.group(), "FILE_PATH"), text
            )
            text = self.windows_path_pattern.sub(
                lambda m: self._generate_consistent_replacement(m.group(), "FILE_PATH"), text
            )
        
        return text
    
    def is_technical_value(self, arg: str) -> bool:
        """Check if an argument is technical and should not be anonymized."""
        if not isinstance(arg, str):
            return True
        
        arg_lower = arg.lower().strip()
        
        number_pattern = r'^-?\d+\.?\d*e?-?\d*$'
        if re.match(number_pattern, arg_lower):
            return True
        
        if arg_lower in ['true', 'false', 'yes', 'no', 'none', 'null']:
            return True
        
        if arg.startswith('--') or re.match(r'^-[a-z]$', arg):
            return True
        
        ml_terms = {
            'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad',
            'bert', 'gpt', 'gpt2', 'gpt3', 'llama', 't5', 'roberta', 'vit', 'clip',
            'resnet', 'resnet50', 'vgg', 'inception', 'efficientnet',
            'lstm', 'gru', 'transformer', 'attention',
            'relu', 'gelu', 'tanh', 'sigmoid', 'softmax', 'elu', 'selu', 'swish',
            'cosine', 'linear', 'polynomial', 'constant', 'exponential',
            'fp16', 'fp32', 'bf16', 'mixed', 'float16', 'float32',
            'cuda', 'cpu', 'gpu', 'mps', 'tpu',
            'train', 'eval', 'test', 'validation', 'inference',
        }
        
        if arg_lower in ml_terms:
            return True
        
        config_extensions = ['.json', '.yaml', '.yml', '.txt', '.csv', '.pth', '.pt', '.ckpt', '.h5', '.pkl']
        if any(arg_lower.endswith(ext) for ext in config_extensions):
            if '/' not in arg and '\\' not in arg and '@' not in arg:
                personal_names = ['user', 'home', 'john', 'alice', 'bob', 'sarah']
                if not any(name in arg_lower for name in personal_names):
                    return True
        
        return False
    
    def anonymize_args(self, args):
        """Anonymize command-line arguments while preserving hyperparameters."""
        if not args:
            return args
        
        if isinstance(args, list):
            result = []
            hyperparam_flags = [
                '--batch_size', '--batch-size', '--batchsize',
                '--learning_rate', '--learning-rate', '--lr',
                '--epochs', '--num_epochs', '--num-epochs',
                '--hidden_size', '--hidden-size', '--hidden_dim',
                '--num_layers', '--num-layers', '--n_layers',
                '--num_heads', '--num-heads', '--n_heads',
                '--dropout', '--dropout_rate', '--dropout-rate',
                '--weight_decay', '--weight-decay',
                '--warmup_steps', '--warmup-steps',
                '--max_length', '--max-length', '--max_len',
                '--seed', '--random_seed', '--random-seed',
                '--gradient_accumulation_steps',
                '--model', '--model_name', '--architecture',
                '--optimizer', '--loss', '--activation',
                '--scheduler', '--lr_scheduler',
                '--precision', '--dtype',
                '--device', '--num_workers', '--workers',
            ]
            
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    if self.is_technical_value(arg):
                        result.append(arg)
                    elif i > 0 and isinstance(result[i-1], str) and result[i-1].startswith('--'):
                        prev_flag = result[i-1].lower()
                        is_hyperparam = any(prev_flag.startswith(hp) for hp in hyperparam_flags)
                        if is_hyperparam:
                            result.append(arg)
                        else:
                            result.append(self.analyze_and_anonymize_text(arg))
                    else:
                        result.append(self.analyze_and_anonymize_text(arg))
                else:
                    result.append(arg)
            return result
        elif isinstance(args, str):
            return self.analyze_and_anonymize_text(args)
        else:
            return args
    
    def anonymize_git_info(self, git_obj):
        """Anonymize git information while preserving commit hashes."""
        if not git_obj or not isinstance(git_obj, dict):
            return git_obj
        
        result = {}
        for key, value in git_obj.items():
            if key in ['commit', 'sha', 'hash', 'commit_hash']:
                result[key] = value
            elif key in ['branch', 'tag', 'ref']:
                if value and isinstance(value, str):
                    value_lower = value.lower()
                    if value_lower in ['main', 'master', 'develop', 'dev', 'staging', 
                                      'production', 'prod', 'release', 'hotfix', 'feature']:
                        result[key] = value
                    elif '/' in value:
                        parts = value.split('/')
                        parts[0] = self.analyze_and_anonymize_text(parts[0])
                        result[key] = '/'.join(parts)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            else:
                if value and isinstance(value, str):
                    result[key] = self.analyze_and_anonymize_text(value)
                else:
                    result[key] = self.anonymize_json_object(value)
        
        return result
    
    def anonymize_json_object(self, obj: Any) -> Any:
        """Recursively anonymize a JSON object using Presidio."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == 'args':
                    result[key] = self.anonymize_args(value)
                elif key == 'git':
                    result[key] = self.anonymize_git_info(value)
                else:
                    result[key] = self.anonymize_json_object(value)
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
                    data = json.load(f)
                    anonymized_data = self.anonymize_json_object(data)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(anonymized_data, out_f, indent=2, ensure_ascii=False)
                
                elif input_path.suffix.lower() == '.jsonl':
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                anonymized_data = self.anonymize_json_object(data)
                                out_f.write(json.dumps(anonymized_data, ensure_ascii=False) + '\n')
                
                elif input_path.suffix.lower() in ['.yaml', '.yml']:
                    # Handle YAML files - anonymize specific fields only, preserve keys
                    import yaml
                    try:
                        data = yaml.safe_load(f)
                        anonymized_data = self.anonymize_yaml_object(data)
                        
                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            yaml.dump(anonymized_data, out_f, default_flow_style=False, allow_unicode=True)
                    except Exception as e:
                        print(f"Warning: Could not parse YAML, treating as text: {e}")
                        f.seek(0)
                        content = f.read()
                        anonymized_content = self.analyze_and_anonymize_text(content)
                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            out_f.write(anonymized_content)
                
                else:
                    # Handle text files (logs, etc.)
                    content = f.read()
                    anonymized_content = self.analyze_and_anonymize_text(content)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(anonymized_content)
        
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
            raise
    
    def anonymize_yaml_object(self, obj: Any) -> Any:
        """
        Anonymize YAML objects selectively with Presidio.
        IMPORTANT: Only anonymize VALUES, never keys (field names).
        Only anonymize fields that likely contain PII (paths, names, emails).
        Preserve technical configuration (hyperparameters, package names, etc.).
        """
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # NEVER anonymize the key itself - preserve field names
                # Only anonymize values in specific fields that may contain PII
                if key in ['prefix', 'name', 'user', 'username', 'author', 'email', 'host', 'hostname']:
                    if isinstance(value, str):
                        result[key] = self.analyze_and_anonymize_text(value)
                    else:
                        result[key] = value
                # Preserve other keys but recursively process nested structures
                else:
                    result[key] = self.anonymize_yaml_object(value)
            return result
        
        elif isinstance(obj, list):
            return [self.anonymize_yaml_object(item) for item in obj]
        
        elif isinstance(obj, str):
            # Only anonymize strings that look like paths or emails in values
            # Don't anonymize technical strings like package names (numpy==1.24.0)
            if '@' in obj and '==' not in obj:  # Email but not package spec
                return self.analyze_and_anonymize_text(obj)
            elif '/' in obj or '\\' in obj:  # Paths
                # Check if it's a package path (like site-packages) vs user path
                if '/home/' in obj or '/Users/' in obj or 'C:\\Users\\' in obj:
                    return self.analyze_and_anonymize_text(obj)
            return obj
        
        else:
            return obj
    
    def anonymize_directory(self, input_dir: str, output_dir: str = None):
        """Anonymize all supported files in a directory using Presidio."""
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir.parent / f"anon_{input_dir.name}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process JSON, JSONL, YAML, and LOG files
        supported_extensions = {'.json', '.jsonl', '.yaml', '.yml', '.log', '.txt'}
        
        print(f"🔍 Scanning directory: {input_dir}")
        files_processed = 0
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
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
                
                results = self.analyzer.analyze(text=text, entities=self.entity_types, language='en')
                
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
    parser = argparse.ArgumentParser(description="Anonymize W&B logs using Microsoft Presidio PII detection")
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
    
    if args.stats and PRESIDIO_AVAILABLE:
        print("🔍 Analyzing PII in input files...")
        stats = anonymizer.get_pii_statistics(args.input)
        print("📊 PII Detection Statistics:")
        for entity_type, count in sorted(stats.items()):
            print(f"   {entity_type}: {count} instances")
        print()
    
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
