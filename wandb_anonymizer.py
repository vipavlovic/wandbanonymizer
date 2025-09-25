#!/usr/bin/env python3
"""
W&B Log Anonymizer Script

This script anonymizes Weights & Biases log files by replacing sensitive information
with anonymized versions while preserving the structure and relationships in the data.
"""

import json
import re
import hashlib
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Set
import uuid

class WandBAnonymizer:
    def __init__(self, seed: str = "default_seed"):
        """
        Initialize the anonymizer with a consistent seed for reproducible anonymization.
        
        Args:
            seed: Seed for consistent hash-based anonymization
        """
        self.seed = seed
        self.username_map = {}
        self.project_map = {}
        self.entity_map = {}
        self.run_id_map = {}
        self.email_map = {}
        self.path_map = {}
        
        # Patterns to identify sensitive information
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.path_pattern = re.compile(r'(/[^/\s]+)+/?')  # Unix paths
        self.windows_path_pattern = re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*')
        
    def _generate_anonymous_name(self, original: str, prefix: str = "") -> str:
        """Generate a consistent anonymous name based on the original."""
        hash_obj = hashlib.md5(f"{self.seed}:{original}".encode())
        hash_hex = hash_obj.hexdigest()[:8]
        return f"{prefix}anon_{hash_hex}" if prefix else f"anon_{hash_hex}"
    
    def anonymize_username(self, username: str) -> str:
        """Anonymize usernames consistently."""
        if username not in self.username_map:
            self.username_map[username] = self._generate_anonymous_name(username, "user_")
        return self.username_map[username]
    
    def anonymize_project(self, project: str) -> str:
        """Anonymize project names consistently."""
        if project not in self.project_map:
            self.project_map[project] = self._generate_anonymous_name(project, "proj_")
        return self.project_map[project]
    
    def anonymize_entity(self, entity: str) -> str:
        """Anonymize entity names consistently."""
        if entity not in self.entity_map:
            self.entity_map[entity] = self._generate_anonymous_name(entity, "entity_")
        return self.entity_map[entity]
    
    def anonymize_run_id(self, run_id: str) -> str:
        """Anonymize run IDs consistently."""
        if run_id not in self.run_id_map:
            # Generate a new UUID-like string but deterministically
            hash_obj = hashlib.md5(f"{self.seed}:run:{run_id}".encode())
            new_uuid = str(uuid.UUID(bytes=hash_obj.digest()))
            self.run_id_map[run_id] = new_uuid
        return self.run_id_map[run_id]
    
    def anonymize_email(self, email: str) -> str:
        """Anonymize email addresses."""
        if email not in self.email_map:
            username, domain = email.split('@', 1)
            anon_username = self._generate_anonymous_name(username)
            anon_domain = "example.com"  # Use a standard anonymous domain
            self.email_map[email] = f"{anon_username}@{anon_domain}"
        return self.email_map[email]
    
    def anonymize_path(self, path: str) -> str:
        """Anonymize file paths while preserving structure."""
        if path not in self.path_map:
            # Split path into components
            path_obj = Path(path)
            parts = path_obj.parts
            
            # Anonymize each part except common system directories
            system_dirs = {'/', 'home', 'usr', 'opt', 'var', 'tmp', 'etc', 'C:', 'Users', 'Program Files'}
            anon_parts = []
            
            for part in parts:
                if part in system_dirs or part.startswith('.'):
                    anon_parts.append(part)
                else:
                    anon_parts.append(self._generate_anonymous_name(part, "dir_"))
            
            self.path_map[path] = str(Path(*anon_parts)) if anon_parts else path
        return self.path_map[path]
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize text content by replacing emails and paths."""
        if not isinstance(text, str):
            return text
            
        # Replace emails
        text = self.email_pattern.sub(lambda m: self.anonymize_email(m.group()), text)
        
        # Replace file paths (be careful not to break URLs)
        if not text.startswith(('http://', 'https://')):
            text = self.path_pattern.sub(lambda m: self.anonymize_path(m.group()), text)
            text = self.windows_path_pattern.sub(lambda m: self.anonymize_path(m.group()), text)
        
        return text
    
    def anonymize_json_object(self, obj: Any) -> Any:
        """Recursively anonymize a JSON object."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Anonymize known sensitive keys
                if key in ['username', 'user', 'author']:
                    result[key] = self.anonymize_username(str(value)) if value else value
                elif key in ['project', 'project_name']:
                    result[key] = self.anonymize_project(str(value)) if value else value
                elif key in ['entity', 'team']:
                    result[key] = self.anonymize_entity(str(value)) if value else value
                elif key in ['id', 'run_id', 'runId']:
                    result[key] = self.anonymize_run_id(str(value)) if value else value
                elif key in ['email', 'email_address']:
                    result[key] = self.anonymize_email(str(value)) if value else value
                elif 'path' in key.lower() or 'dir' in key.lower():
                    result[key] = self.anonymize_path(str(value)) if value else value
                else:
                    # Recursively process the value
                    result[key] = self.anonymize_json_object(value)
            return result
        
        elif isinstance(obj, list):
            return [self.anonymize_json_object(item) for item in obj]
        
        elif isinstance(obj, str):
            return self.anonymize_text(obj)
        
        else:
            return obj
    
    def anonymize_file(self, input_path: str, output_path: str = None):
        """Anonymize a W&B log file."""
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"anon_{input_path.name}"
        else:
            output_path = Path(output_path)
        
        print(f"Anonymizing {input_path} -> {output_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.suffix.lower() == '.json':
                    # Handle JSON files
                    data = json.load(f)
                    anonymized_data = self.anonymize_json_object(data)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(anonymized_data, out_f, indent=2, ensure_ascii=False)
                
                elif input_path.suffix.lower() == '.jsonl':
                    # Handle JSONL files (common for W&B logs)
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                anonymized_data = self.anonymize_json_object(data)
                                out_f.write(json.dumps(anonymized_data, ensure_ascii=False) + '\n')
                
                else:
                    # Handle text files
                    content = f.read()
                    anonymized_content = self.anonymize_text(content)
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(anonymized_content)
        
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            raise
    
    def anonymize_directory(self, input_dir: str, output_dir: str = None):
        """Anonymize all supported files in a directory."""
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir.parent / f"anon_{input_dir.name}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions commonly found in W&B logs
        supported_extensions = {'.json', '.jsonl', '.txt', '.log', '.yaml', '.yml'}
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Maintain directory structure
                relative_path = file_path.relative_to(input_dir)
                output_file_path = output_dir / relative_path
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.anonymize_file(file_path, output_file_path)
    
    def save_mapping(self, output_path: str):
        """Save the anonymization mapping to a file (for reference, keep secure!)."""
        mapping = {
            'usernames': self.username_map,
            'projects': self.project_map,
            'entities': self.entity_map,
            'run_ids': self.run_id_map,
            'emails': self.email_map,
            'paths': self.path_map
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Anonymization mapping saved to {output_path}")
        print("WARNING: Keep this mapping file secure as it contains the original->anonymous mappings!")


def main():
    parser = argparse.ArgumentParser(description="Anonymize Weights & Biases log files")
    parser.add_argument("input", help="Input file or directory to anonymize")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("-s", "--seed", default="wandb_anon_2024", 
                       help="Seed for consistent anonymization")
    parser.add_argument("-m", "--save-mapping", 
                       help="Save anonymization mapping to specified file")
    
    args = parser.parse_args()
    
    anonymizer = WandBAnonymizer(seed=args.seed)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        anonymizer.anonymize_file(args.input, args.output)
    elif input_path.is_dir():
        anonymizer.anonymize_directory(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    if args.save_mapping:
        anonymizer.save_mapping(args.save_mapping)
    
    print("Anonymization complete!")
    return 0


if __name__ == "__main__":
    exit(main())
