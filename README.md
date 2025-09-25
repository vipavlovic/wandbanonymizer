# W&B Log Anonymizer

A comprehensive tool for anonymizing Weights & Biases (W&B) log files while preserving data relationships and structure. This tool helps you safely share ML experiment data by replacing sensitive information with consistent anonymous identifiers.

## 🚀 Quick Start

```bash
# Clone or download the scripts
# Run the complete test suite
python test_anonymization.py

# Anonymize your own W&B logs
python wandb_anonymizer.py /path/to/your/wandb/logs -o /path/to/anonymized/logs
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `wandb_anonymizer.py` | **Main anonymization script** - Core tool for anonymizing W&B logs |
| `test_anonymization.py` | **Automated test suite** - Tests the anonymizer with sample data |
| `generate_sample_logs.py` | **Basic test data generator** - Creates simple W&B log samples |
| `realistic_wandb_logs.py` | **Realistic test data generator** - Creates real-world-like W&B logs |

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- Standard libraries only (no additional dependencies required)

### Setup
1. Download all four Python files to the same directory
2. Make sure they're executable:
   ```bash
   chmod +x *.py
   ```

## 📋 Usage

### Basic Usage

**Anonymize a single file:**
```bash
python wandb_anonymizer.py wandb-metadata.json
# Output: anon_wandb-metadata.json
```

**Anonymize with custom output:**
```bash
python wandb_anonymizer.py wandb-metadata.json -o anonymized_metadata.json
```

**Anonymize entire directory:**
```bash
python wandb_anonymizer.py ./wandb-logs/ -o ./anonymized-logs/
```

**Save anonymization mapping (keep secure!):**
```bash
python wandb_anonymizer.py ./wandb-logs/ -o ./anonymized-logs/ -m mapping.json
```

### Advanced Usage

**Use custom seed for reproducible anonymization:**
```bash
python wandb_anonymizer.py ./wandb-logs/ -s "my-secret-seed-2024"
```

**Full command with all options:**
```bash
python wandb_anonymizer.py ./wandb-logs/ \
  --output ./anonymized-logs/ \
  --seed "my-project-seed" \
  --save-mapping ./mapping.json
```

### Command Line Options

```
python wandb_anonymizer.py <input> [options]

Arguments:
  input                 Input file or directory to anonymize

Options:
  -o, --output         Output file or directory path
  -s, --seed          Seed for consistent anonymization (default: wandb_anon_2024)
  -m, --save-mapping  Save anonymization mapping to specified file
  -h, --help          Show help message
```

## 🧪 Testing

### Run Complete Test Suite
```bash
python test_anonymization.py
```

This will:
1. Generate sample W&B logs
2. Generate realistic W&B logs (based on real ML projects)
3. Test anonymization on single files
4. Test anonymization on directories
5. Create mapping files
6. Compare original vs anonymized content

### Generate Test Data Only
```bash
# Basic sample logs
python generate_sample_logs.py

# Realistic logs (based on KoAlpaca project)
python realistic_wandb_logs.py
```

## 🔒 What Gets Anonymized

### Personal Information
- **Usernames**: `john.doe` → `user_anon_a1b2c3d4`
- **Email addresses**: `user@company.com` → `anon_xyz123@example.com`
- **Entity/team names**: `acme-corp` → `entity_anon_e4f5g6h7`

### System Information
- **File paths**: `/home/user/project/` → `/dir_anon_i8j9k0l1/dir_anon_m2n3o4p5/`
- **Hostnames**: `user-workstation.company.com` → Anonymized consistently
- **Run IDs**: Original UUIDs → New consistent UUIDs

### Project Information
- **Project names**: `my-ml-project` → `proj_anon_q6r7s8t9`
- **Git repositories**: Repository URLs anonymized
- **Code paths**: Local file paths anonymized

## 📊 Supported File Formats

- **JSON files** (`.json`) - W&B metadata, configs, summaries
- **JSONL files** (`.jsonl`) - W&B history logs, events
- **Text files** (`.txt`, `.log`) - Output logs, debug files
- **YAML files** (`.yaml`, `.yml`) - Configuration files

## 🛡️ Security Features

### Consistent Anonymization
- Same original value always maps to same anonymous value
- Preserves relationships between runs and experiments
- Deterministic based on seed (reproducible)

### Data Integrity
- **Structure preserved**: JSON structure, data types maintained
- **Relationships maintained**: Links between files preserved
- **Metrics unchanged**: Numerical values, timestamps, model metrics unchanged

### Reversible Mapping
- Optional mapping file shows `original → anonymous`
- **⚠️ Keep mapping file secure** - it can reverse anonymization
- Useful for internal reference and debugging

## 📝 Examples

### Example 1: Anonymize W&B Run Directory
```bash
# Your original W&B run directory
ls wandb/run-20240101_123456-abc123/files/
# wandb-metadata.json  config.yaml  wandb-history.jsonl  wandb-summary.json

# Anonymize it
python wandb_anonymizer.py wandb/run-20240101_123456-abc123/files/ -o anonymized_run/

# Check results
ls anonymized_run/
# wandb-metadata.json  config.yaml  wandb-history.jsonl  wandb-summary.json (anonymized)
```

### Example 2: Before/After Comparison

**Original wandb-metadata.json:**
```json
{
  "username": "john.doe",
  "email": "john.doe@company.com", 
  "host": "johns-workstation.company.com",
  "root": "/home/john/ml-project",
  "program": "/home/john/ml-project/train.py"
}
```

**Anonymized wandb-metadata.json:**
```json
{
  "username": "user_anon_a1b2c3d4",
  "email": "anon_e5f6g7h8@example.com",
  "host": "anon_i9j0k1l2", 
  "root": "/dir_anon_m3n4o5p6/dir_anon_q7r8s9t0",
  "program": "/dir_anon_m3n4o5p6/dir_anon_q7r8s9t0/anon_u1v2w3x4"
}
```

## ⚠️ Important Security Notes

### 🔐 Mapping File Security
The mapping file contains original → anonymous mappings:
```json
{
  "usernames": {"john.doe": "user_anon_a1b2c3d4"},
  "emails": {"john.doe@company.com": "anon_e5f6g7h8@example.com"}
}
```
**Keep this file secure** - treat it like the original sensitive data!

### 🎯 Seed Consistency
- Use the same seed when anonymizing related files
- Different seeds = different anonymous identifiers
- Store your seed securely for reproducibility

### 🔍 Manual Review
- Always review anonymized files before sharing
- Check for any missed sensitive information
- Verify that data relationships are preserved

## 🐛 Troubleshooting

### Common Issues

**Permission denied:**
```bash
chmod +x wandb_anonymizer.py
```

**File encoding errors:**
```bash
# The script handles UTF-8 encoding automatically
# For other encodings, convert first:
iconv -f ISO-8859-1 -t UTF-8 input.json > input_utf8.json
```

**Large files taking too long:**
- Process files individually instead of entire directories
- Use faster storage (SSD) if processing many files

### Debug Mode
Add print statements to see what's being processed:
```python
# In wandb_anonymizer.py, add:
print(f"Processing: {input_path}")
```

## 🤝 Contributing

### Found a bug?
1. Check if sensitive data is still visible in anonymized files
2. Test with the provided test suite
3. Report issues with sample data (not real sensitive data!)

### Want to add features?
- Support for additional file formats
- More anonymization patterns
- Performance improvements
- Better error handling

## 📄 License

This tool is provided as-is for educational and research purposes. Use responsibly and ensure compliance with your organization's data privacy policies.

## 📚 Additional Resources

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Data Export Guide](https://docs.wandb.ai/guides/track/public-api-guide/)
- [Data Privacy Best Practices](https://docs.wandb.ai/guides/hosting/security/)

---

**🚨 Remember: This tool helps anonymize data, but always review the output before sharing. No tool is perfect - manual verification is essential for sensitive data.**