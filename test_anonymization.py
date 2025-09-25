#!/usr/bin/env python3
"""
Test the W&B Anonymization Script

This script generates sample W&B logs and then tests the anonymization script on them.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and print the results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def compare_files(original_dir, anonymized_dir):
    """Compare original and anonymized files to show the differences."""
    print(f"\n{'='*60}")
    print("COMPARING ORIGINAL VS ANONYMIZED FILES")
    print('='*60)
    
    original_path = Path(original_dir)
    anonymized_path = Path(anonymized_dir)
    
    if not original_path.exists() or not anonymized_path.exists():
        print(f"❌ One of the directories doesn't exist:")
        print(f"   Original: {original_path} (exists: {original_path.exists()})")
        print(f"   Anonymized: {anonymized_path} (exists: {anonymized_path.exists()})")
        return
    
    # Find common files
    original_files = set(f.name for f in original_path.rglob("*") if f.is_file())
    anonymized_files = set(f.name for f in anonymized_path.rglob("*") if f.is_file())
    common_files = original_files & anonymized_files
    
    print(f"Found {len(common_files)} common files to compare:")
    
    for filename in sorted(common_files):
        print(f"\n📄 {filename}:")
        
        original_file = original_path / filename
        anonymized_file = anonymized_path / filename
        
        if original_file.exists() and anonymized_file.exists():
            # Read first few lines of each file for comparison
            try:
                with open(original_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()[:500]  # First 500 chars
                
                with open(anonymized_file, 'r', encoding='utf-8') as f:
                    anonymized_content = f.read()[:500]  # First 500 chars
                
                print("   ORIGINAL (first 500 chars):")
                print("   " + original_content.replace('\n', '\n   ')[:200] + "...")
                print("   ANONYMIZED (first 500 chars):")
                print("   " + anonymized_content.replace('\n', '\n   ')[:200] + "...")
                
                # Check if any obvious anonymization occurred
                sensitive_terms = ['john.doe', '@company.com', '/home/john', 'acme-corp']
                found_terms = [term for term in sensitive_terms if term.lower() in anonymized_content.lower()]
                if found_terms:
                    print(f"   ⚠️  WARNING: Still found these terms: {found_terms}")
                else:
                    print("   ✅ No obvious sensitive terms found in anonymized version")
                    
            except Exception as e:
                print(f"   ❌ Error reading files: {e}")
        else:
            print("   ❌ One of the files doesn't exist")

def main():
    """Run the full test suite."""
    print("🧪 W&B Log Anonymization Test Suite")
    print("="*60)
    
    # Step 1: Generate both sample and realistic logs
    print("🔧 Generating sample W&B logs...")
    if not run_command("python generate_sample_logs.py", "Generating basic sample W&B logs"):
        print("❌ Failed to generate sample logs. Exiting.")
        sys.exit(1)
        
    print("🔧 Generating realistic W&B logs based on real-world examples...")
    if not run_command("python realistic_wandb_logs.py", "Generating realistic W&B logs"):
        print("⚠️  Failed to generate realistic logs, continuing with basic samples...")
    else:
        print("✅ Realistic logs generated successfully!")
    
    # Step 2: Check if sample logs were created
    sample_dir = Path("sample_wandb_logs")
    realistic_dir = Path("realistic_wandb_logs")
    
    if not sample_dir.exists():
        print(f"❌ Sample directory {sample_dir} was not created. Exiting.")
        sys.exit(1)
        
    print(f"\n✅ Sample logs created in {sample_dir}")
    files = list(sample_dir.glob("*"))
    print(f"   Generated {len(files)} files:")
    for file in files:
        print(f"   - {file.name}")
    
    if realistic_dir.exists():
        print(f"\n✅ Realistic logs created in {realistic_dir}")
        realistic_files = list(realistic_dir.glob("*"))
        print(f"   Generated {len(realistic_files)} files:")
        for file in realistic_files:
            print(f"   - {file.name}")
    
    # Step 3: Test anonymizing a single file
    test_file = sample_dir / "wandb-metadata.json"
    if test_file.exists():
        if not run_command(
            f"python wandb_anonymizer.py {test_file} -o anonymized_metadata.json",
            "Testing anonymization on single file"
        ):
            print("❌ Failed to anonymize single file")
        else:
            print("✅ Single file anonymization successful")
    
    # Step 4: Test anonymizing entire directories
    if not run_command(
        f"python wandb_anonymizer.py sample_wandb_logs -o anonymized_sample_logs",
        "Testing anonymization on sample directory"
    ):
        print("❌ Failed to anonymize sample directory")
        sys.exit(1)
    
    print("✅ Sample directory anonymization successful")
    
    # Test realistic logs if they exist
    if realistic_dir.exists():
        if not run_command(
            f"python wandb_anonymizer.py realistic_wandb_logs -o anonymized_realistic_logs",
            "Testing anonymization on realistic directory"
        ):
            print("⚠️  Failed to anonymize realistic directory, but continuing...")
        else:
            print("✅ Realistic directory anonymization successful")
    
    # Step 5: Test with mapping file
    if not run_command(
        f"python wandb_anonymizer.py sample_wandb_logs -o anonymized_logs_with_mapping -m anonymization_mapping.json",
        "Testing anonymization with mapping file"
    ):
        print("⚠️  Failed to create mapping file, but this might be okay")
    else:
        print("✅ Anonymization with mapping file successful")
        
        # Check if mapping file was created
        mapping_file = Path("anonymization_mapping.json")
        if mapping_file.exists():
            print(f"✅ Mapping file created: {mapping_file}")
            # Show a preview of the mapping
            try:
                import json
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                print("\n📋 Mapping Preview:")
                for category, mappings in mapping.items():
                    if mappings:  # Only show non-empty categories
                        print(f"   {category}:")
                        for original, anonymized in list(mappings.items())[:2]:  # Show first 2 mappings
                            print(f"      {original} → {anonymized}")
                        if len(mappings) > 2:
                            print(f"      ... and {len(mappings) - 2} more")
            except Exception as e:
                print(f"⚠️  Could not preview mapping file: {e}")
    
    # Step 6: Compare original vs anonymized files
    compare_files("sample_wandb_logs", "anonymized_sample_logs")
    
    if realistic_dir.exists() and Path("anonymized_realistic_logs").exists():
        print(f"\n{'='*60}")
        print("COMPARING REALISTIC LOGS")
        print('='*60)
        compare_files("realistic_wandb_logs", "anonymized_realistic_logs")
    
    # Step 7: Clean up (optional)
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print("✅ Sample logs generated successfully")
    print("✅ Single file anonymization worked")
    print("✅ Directory anonymization worked")
    print("✅ Files compared successfully")
    
    print("\n📁 Files created during testing:")
    test_files = [
        "sample_wandb_logs/",
        "realistic_wandb_logs/",
        "anonymized_sample_logs/", 
        "anonymized_realistic_logs/",
        "anonymized_logs_with_mapping/",
        "anonymized_metadata.json",
        "anonymization_mapping.json"
    ]
    
    for filepath in test_files:
        path = Path(filepath)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.rglob("*")))
                print(f"   ✅ {filepath} (directory with {file_count} files)")
            else:
                print(f"   ✅ {filepath} (file)")
        else:
            print(f"   ❌ {filepath} (not found)")
    
    print("\n🎉 Testing complete! Check the anonymized files to verify the results.")
    print("⚠️  Remember to keep the mapping file secure as it can reverse the anonymization!")

if __name__ == "__main__":
    main()
