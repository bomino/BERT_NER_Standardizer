# NER Tagger for Hardware Item Descriptions
## Documentation

### Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Guide](#usage-guide)
5. [Configuration](#configuration)
6. [Input Formats](#input-formats)
7. [Output Format](#output-format)
8. [Entity Types](#entity-types)
9. [Statistics and Reporting](#statistics-and-reporting)
10. [Error Handling](#error-handling)
11. [Advanced Usage](#advanced-usage)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

### Overview
The NER (Named Entity Recognition) Tagger is a specialized tool for processing hardware and construction material descriptions. It identifies and labels various components of item descriptions using BIO (Beginning, Inside, Outside) tagging scheme. The tool is particularly designed for processing inventory items, product catalogs, and material lists.

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ner-tagger.git
cd ner-tagger

# Ensure you have Python 3.6+
python --version

# No additional packages required - uses standard library only
```

### Quick Start
```bash
# Basic usage
python ner_tagger.py input.txt

# Save output to file
python ner_tagger.py input.txt -o tagged_items.py

# Show statistics
python ner_tagger.py input.txt --stats

# Use custom configuration
python ner_tagger.py input.txt --config custom_config.yaml --stats
```

### Usage Guide

#### Command Line Arguments
- `input_file`: Path to input file (required)
- `-o, --output`: Path to output file (optional)
- `--config`: Path to configuration file (optional)
- `--stats`: Show tagging statistics (optional)
- `--debug`: Enable debug logging (optional)

#### Example Commands
```bash
# Process items and show statistics
python ner_tagger.py catalog.csv --stats

# Use custom configuration and save output
python ner_tagger.py items.txt --config hardware_config.yaml -o results.py

# Debug mode with statistics
python ner_tagger.py input.csv --debug --stats
```

### Configuration

#### Default Configuration
The script comes with default configurations for:
- Unit aliases (IN → INCH, LB → POUND, etc.)
- Brand names
- Product types
- Specifications
- Measurement patterns

#### Custom Configuration
Create a YAML or JSON file to override defaults:

```yaml
# custom_config.yaml
unit_aliases:
  INCHES: "IN"
  FEET: "FT"
  POUNDS: "LB"

brands:
  - DEWALT
  - MILWAUKEE
  - MAKITA

types:
  - DRILL
  - SAW
  - HAMMER

specs:
  - HEAVY
  - DUTY
  - PREMIUM
```

### Input Formats

#### Text File (.txt)
```text
11737 5/8-4-8 TYPE X DRYWALL
42418 PURDY 2.5-IN XL GLIDE ANG
ITEM # 476043 TAPCON 1/4-IN X 7-IN SDS DRILLBIT
```

#### CSV File (.csv)
```csv
Item Description,SKU,Price
11737 5/8-4-8 TYPE X DRYWALL,SKU123,$10.99
42418 PURDY 2.5-IN XL GLIDE ANG,SKU456,$15.99
```

### Output Format
```python
labeled_data = [
    {
     "tokens": ["11737", "5/8", "-", "4", "-", "8", "TYPE", "X", "DRYWALL"],
     "ner_tags": ["B-ITEMNUM", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", 
                  "B-SPEC", "I-SPEC", "B-TYPE"]
    }
]
```

### Entity Types

#### ITEMNUM (Item Numbers)
- Format: Usually at start of description
- Examples: "11737", "42418", "476043"
- Patterns: Numeric, alphanumeric

#### BRAND (Brand Names)
- Examples: "DEWALT", "MILWAUKEE", "PURDY"
- Can be multi-token: "BLACK AND DECKER"

#### SIZE (Measurements)
- Formats: 
  - Simple: "5/8", "2.5"
  - With units: "2.5-IN", "10-FT"
  - Dimensions: "2X4X8"

#### SPEC (Specifications)
- Types:
  - Grades: "TYPE X", "GRADE A"
  - Properties: "HEAVY DUTY"
  - Performance: "FAST DRY"

#### TYPE (Product Types)
- Categories:
  - Materials: "DRYWALL", "LUMBER"
  - Tools: "DRILL", "SAW"
  - Parts: "SCREW", "NAIL"

#### QUANTITY
- Format: Number + Unit
- Examples: "100 CT", "5 LB", "12 PC"

### Statistics and Reporting

#### Basic Statistics
```
=== Tagging Statistics Report ===
Processed 150 items with 1200 tokens

Tag Distribution:
B-ITEMNUM     150  (10.5%)
B-BRAND       145   (9.8%)
B-SIZE        200  (13.5%)
```

#### Entity Length Statistics
```
Entity Length Statistics:
BRAND         Avg: 1.2  Max: 3
TYPE          Avg: 1.5  Max: 4
SPEC          Avg: 2.1  Max: 5
```

### Error Handling

#### Common Errors
1. Input Validation
   ```python
   ValidationError: Item description too long: ...
   ValidationError: Empty item description
   ```

2. Tokenization
   ```python
   TokenizationError: Empty token found in sequence
   TokenizationError: Invalid character in token
   ```

3. Configuration
   ```python
   ConfigurationError: Unsupported config file format
   ConfigurationError: Invalid configuration value
   ```

### Advanced Usage

#### Custom Entity Rules
```python
# Add custom rules in configuration
custom_rules:
  brand_patterns:
    - "[A-Z]+-[A-Z]+"
    - "[A-Z]+PRO"
  size_patterns:
    - "\d+(/\d+)?-[A-Z]{2}"
```

#### Processing Pipeline
1. Input Validation
2. Tokenization
3. Entity Detection
4. BIO Tagging
5. Statistics Collection
6. Output Generation

### Best Practices

1. Input Data Preparation
   - Clean up inconsistent formatting
   - Remove special characters
   - Standardize units and measurements

2. Configuration Management
   - Start with default configuration
   - Add custom entities gradually
   - Test configuration changes

3. Output Handling
   - Use meaningful file names
   - Review statistics regularly
   - Monitor error rates

### Troubleshooting

#### Common Issues

1. Incorrect Tagging
   - Check entity lists in configuration
   - Verify input format
   - Review similar correct examples

2. Performance Issues
   - Reduce input file size
   - Remove unnecessary whitespace
   - Clean input data

3. Configuration Problems
   - Validate YAML/JSON syntax
   - Check for duplicate entries
   - Verify file paths

#### Debug Mode
Enable debug mode for detailed logging:
```bash
python ner_tagger.py input.txt --debug
```

This will show:
- Tokenization details
- Entity detection steps
- Configuration loading
- Error traces

### Support and Contributions

#### Reporting Issues
- Provide example input
- Include error messages
- Attach configuration file

#### Making Contributions
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit pull request

#### Contact
For support or questions:
- GitHub Issues
- Documentation Wiki
- Project Maintainers