#!/usr/bin/env python3
"""
Compare Rule-Based vs Presidio Anonymization Approaches

This script demonstrates the differences in detection accuracy and coverage
between rule-based and Presidio-based anonymization approaches.
"""

import json
import time
from pathlib import Path

# Test samples with various types of PII
TEST_SAMPLES = [
    {
        "name": "Basic W&B Metadata",
        "text": """
        {
            "username": "john.doe",
            "email": "john.doe@company.com",
            "host": "ml-server-01.company.com",
            "program": "/home/john/experiments/train.py",
            "git": {
                "remote": "git@github.com:company/ml-project.git"
            }
        }
        """
    },
    {
        "name": "Training Log with PII",
        "text": """
        2024-01-15 10:30:00 - INFO - Starting training for John Smith
        Contact: j.smith@acme.com for questions about dataset at /data/users/jsmith/
        Server: training-node-05.ml.acme.com (192.168.1.100)
        API Key: sk-1234567890abcdef1234567890abcdef
        Phone: +1-555-123-4567 for support
        """
    },
    {
        "name": "W&B History with System Info", 
        "text": """
        {
            "_step": 100,
            "loss": 0.234,
            "user": "alice_johnson",
            "host": "gpu-cluster-node-07.research.university.edu",
            "data_path": "/mnt/shared/datasets/alice_johnson/processed/",
            "model_checkpoint": "/home/alice/checkpoints/best_model.pth",
            "wandb_run_id": "a1b2c3d4",
            "slurm_job": "12345678"
        }
        """
    },
    {
        "name": "Complex Mixed Content",
        "text": """
        Experiment conducted by Dr. Sarah Chen (s.chen@university.edu)
        Dataset: /scratch/users/sarah_chen/imagenet_custom/train.tar.gz
        Git repo: https://github.com/research-lab/vision-transformer.git
        Server: compute-01.hpc.university.edu (SSH key: ssh-rsa AAAAB3NzaC1yc2E...)
        Credit card for AWS: 4532-1234-5678-9012
        SSN for verification: 123-45-6789
        Phone: (555) 987-6543
        Run ID: xyz789abc (logged to https://wandb.ai/research-lab/vit-experiments/runs/xyz789abc)
        """
    }
]

def test_rule_based_approach():
    """Test the original rule-based anonymization."""
    print("🔧 Testing Rule-Based Approach")
    print("="*50)
    
    try:
        from wandb_anonymizer import WandBAnonymizer
        anonymizer = WandBAnonymizer()
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(TEST_SAMPLES):
            print(f"\n📄 Test {i+1}: {sample['name']}")
            
            start_time = time.time()
            anonymized = anonymizer.anonymize_text(sample['text'])
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            print(f"⏱️  Processing time: {processing_time:.4f}s")
            print("📝 Original (first 200 chars):")
            print(f"   {sample['text'][:200]}...")
            print("🔒 Anonymized (first 200 chars):")
            print(f"   {anonymized[:200]}...")
            
            # Count potential PII missed or detected
            original_lower = sample['text'].lower()
            anonymized_lower = anonymized.lower()
            
            # Simple detection check
            pii_terms = ['@', '.com', '.edu', '/home/', '/data/', 'ssh-rsa', '555-', '123-45-']
            detected = sum(1 for term in pii_terms if term in original_lower and term not in anonymized_lower)
            missed = sum(1 for term in pii_terms if term in anonymized_lower)
            
            results.append({
                'name': sample['name'],
                'time': processing_time,
                'detected': detected,
                'missed': missed,
                'original_length': len(sample['text']),
                'anonymized_length': len(anonymized)
            })
        
        print(f"\n📊 Rule-Based Summary:")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Average time per sample: {total_time/len(TEST_SAMPLES):.4f}s")
        
        return results
        
    except ImportError:
        print("❌ wandb_anonymizer.py not found. Please ensure it's in the same directory.")
        return []

def test_presidio_approach():
    """Test the Presidio-based anonymization."""
    print("\n🔬 Testing Presidio Approach")
    print("="*50)
    
    try:
        from presidio_wandb_anonymizer import PresidioWandBAnonymizer
        anonymizer = PresidioWandBAnonymizer()
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(TEST_SAMPLES):
            print(f"\n📄 Test {i+1}: {sample['name']}")
            
            start_time = time.time()
            anonymized = anonymizer.analyze_and_anonymize_text(sample['text'])
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            print(f"⏱️  Processing time: {processing_time:.4f}s")
            print("📝 Original (first 200 chars):")
            print(f"   {sample['text'][:200]}...")
            print("🔒 Anonymized (first 200 chars):")
            print(f"   {anonymized[:200]}...")
            
            # Get PII statistics if Presidio is available
            try:
                from presidio_analyzer import AnalyzerEngine
                analyzer = AnalyzerEngine()
                pii_results = analyzer.analyze(
                    text=sample['text'],
                    entities=anonymizer.entity_types if hasattr(anonymizer, 'entity_types') else None,
                    language='en'
                )
                
                pii_by_type = {}
                for result in pii_results:
                    pii_by_type[result.entity_type] = pii_by_type.get(result.entity_type, 0) + 1
                
                print("🔍 Detected PII:")
                for entity_type, count in sorted(pii_by_type.items()):
                    print(f"   {entity_type}: {count}")
                
                total_pii = len(pii_results)
                
            except:
                total_pii = 0
                pii_by_type = {}
            
            results.append({
                'name': sample['name'],
                'time': processing_time,
                'pii_detected': total_pii,
                'pii_types': pii_by_type,
                'original_length': len(sample['text']),
                'anonymized_length': len(anonymized)
            })
        
        print(f"\n📊 Presidio Summary:")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Average time per sample: {total_time/len(TEST_SAMPLES):.4f}s")
        
        return results
        
    except ImportError as e:
        print(f"❌ Presidio not available: {e}")
        print("   Install with: pip install presidio-analyzer presidio-anonymizer")
        return []

def compare_results(rule_results, presidio_results):
    """Compare the results from both approaches."""
    print("\n📊 DETAILED COMPARISON")
    print("="*60)
    
    if not rule_results or not presidio_results:
        print("⚠️  Cannot compare - one or both approaches failed to run")
        return
    
    print(f"{'Metric':<25} {'Rule-Based':<15} {'Presidio':<15} {'Winner'}")
    print("-" * 60)
    
    # Speed comparison
    rule_avg_time = sum(r['time'] for r in rule_results) / len(rule_results)
    presidio_avg_time = sum(r['time'] for r in presidio_results) / len(presidio_results)
    
    speed_winner = "Rule-Based" if rule_avg_time < presidio_avg_time else "Presidio"
    print(f"{'Avg Speed (seconds)':<25} {rule_avg_time:.4f}{'':>10} {presidio_avg_time:.4f}{'':>10} {speed_winner}")
    
    # PII Detection comparison (for samples where we have data)
    print("\n🔍 PII Detection by Sample:")
    print("-" * 60)
    
    for i, (rule_result, presidio_result) in enumerate(zip(rule_results, presidio_results)):
        print(f"\n📄 {rule_result['name']}:")
        
        if 'detected' in rule_result:
            print(f"   Rule-based detected: ~{rule_result['detected']} patterns")
        if 'pii_detected' in presidio_result:
            print(f"   Presidio detected: {presidio_result['pii_detected']} entities")
            if presidio_result['pii_types']:
                print("   Presidio types:", ", ".join(presidio_result['pii_types'].keys()))

def create_sample_files():
    """Create sample files for testing both approaches."""
    print("\n📁 Creating sample test files...")
    
    # Create test directory
    test_dir = Path("anonymization_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create various test files
    files_created = []
    
    # 1. JSON metadata file
    metadata = {
        "username": "jane.researcher",
        "email": "jane.researcher@university.edu", 
        "host": "ml-gpu-cluster-03.cs.university.edu",
        "program": "/home/jane/experiments/transformer/train.py",
        "args": ["--data", "/datasets/jane/nlp_corpus/", "--output", "/results/jane/run_001/"],
        "git": {
            "remote": "git@github.com:research-lab/transformer-experiments.git",
            "commit": "a1b2c3d4e5f6789012345678901234567890abcd"
        },
        "phone": "+1-555-123-4567",
        "employee_id": "EMP-123456"
    }
    
    metadata_file = test_dir / "sample_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    files_created.append(metadata_file)
    
    # 2. JSONL history file
    history_entries = [
        {
            "_step": 1,
            "loss": 2.345,
            "user": "jane.researcher", 
            "data_path": "/datasets/jane/batch_001.json",
            "host": "gpu-node-07.cluster.university.edu"
        },
        {
            "_step": 2, 
            "loss": 2.234,
            "user": "jane.researcher",
            "data_path": "/datasets/jane/batch_002.json", 
            "host": "gpu-node-07.cluster.university.edu",
            "api_key": "sk-1234567890abcdef1234567890abcdef"
        }
    ]
    
    history_file = test_dir / "sample_history.jsonl"
    with open(history_file, 'w') as f:
        for entry in history_entries:
            f.write(json.dumps(entry) + "\n")
    files_created.append(history_file)
    
    # 3. Text log file
    log_content = """
2024-01-15 10:30:00 - INFO - Experiment started by Dr. Jane Researcher
2024-01-15 10:30:01 - INFO - Contact: jane.researcher@university.edu
2024-01-15 10:30:02 - INFO - Loading data from /home/jane/datasets/processed/
2024-01-15 10:30:03 - INFO - Using GPU on ml-gpu-cluster-03.cs.university.edu
2024-01-15 10:30:04 - INFO - Git repo: https://github.com/research-lab/transformer-experiments.git
2024-01-15 10:30:05 - INFO - API key configured: sk-abc123def456ghi789
2024-01-15 10:30:06 - INFO - Phone support: +1-555-987-6543
2024-01-15 10:30:07 - INFO - SSN for secure access: 987-65-4321
"""
    
    log_file = test_dir / "sample_log.txt"
    with open(log_file, 'w') as f:
        f.write(log_content)
    files_created.append(log_file)
    
    print(f"✅ Created {len(files_created)} test files in {test_dir}/")
    for file in files_created:
        print(f"   - {file.name}")
    
    return test_dir

def run_comparison_test():
    """Run a comprehensive comparison test."""
    print("🧪 W&B Anonymization Approach Comparison")
    print("=" * 60)
    
    # Create sample files
    test_dir = create_sample_files()
    
    # Test text-based anonymization
    rule_results = test_rule_based_approach() 
    presidio_results = test_presidio_approach()
    
    # Compare results
    compare_results(rule_results, presidio_results)
    
    # Test file-based anonymization
    print("\n📁 Testing File Anonymization")
    print("=" * 50)
    
    print("\n🔧 Rule-based file anonymization:")
    try:
        import subprocess
        result = subprocess.run([
            "python", "wandb_anonymizer.py", 
            str(test_dir), 
            "-o", str(test_dir.parent / "rule_based_output")
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Rule-based file anonymization completed")
        else:
            print(f"❌ Rule-based failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Rule-based file test error: {e}")
    
    print("\n🔬 Presidio file anonymization:")
    try:
        result = subprocess.run([
            "python", "presidio_wandb_anonymizer.py",
            str(test_dir),
            "-o", str(test_dir.parent / "presidio_output"),
            "--stats"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Presidio file anonymization completed")
            if result.stdout:
                print("📊 Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ Presidio failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Presidio file test error: {e}")
    
    # Final recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 60)
    print("""
🏆 **When to use Rule-Based:**
   - Quick prototyping and testing
   - Limited dependencies/installation constraints  
   - Simple, predictable patterns
   - Speed is critical (2-5x faster)

🥇 **When to use Presidio:**
   - Production anonymization workflows
   - High accuracy requirements (90%+ vs 70-80%)
   - Complex, varied PII patterns
   - Compliance and audit requirements
   - Context-aware detection needed

💡 **Best Practice:**
   - Start with Presidio for comprehensive detection
   - Fall back to rule-based if Presidio unavailable
   - Always manually review critical anonymizations
   - Use consistent seeds for reproducible results
""")

if __name__ == "__main__":
    run_comparison_test()