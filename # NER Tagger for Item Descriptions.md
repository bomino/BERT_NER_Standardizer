# NER Tagger for Item Descriptions
## Documentation

### Overview
The NER (Named Entity Recognition) Tagger is a Python script that processes item descriptions and generates BIO (Beginning, Inside, Outside) tags for various entities such as brands, sizes, types, specifications, and quantities. It's specifically designed for processing hardware and construction material descriptions.

### Requirements
- Python 3.6 or higher
- Standard Python libraries (no additional installations required):
  - `argparse`
  - `csv`
  - `pathlib`
  - `re`
  - `sys`

### Installation
1. Download the script `ner_tagger.py`
2. Ensure you have Python 3.6+ installed
3. No additional package installation is needed

### Basic Usage
```bash
python ner_tagger.py input_file.txt
```

### Command Line Arguments
```bash
python ner_tagger.py [-h] [-o OUTPUT] [--stats] input_file
```

#### Required Arguments:
- `input_file`: Path to the input file containing item descriptions

#### Optional Arguments:
- `-h, --help`: Show help message and exit
- `-o OUTPUT, --output OUTPUT`: Path to output file (if not specified, prints to stdout)
- `--stats`: Show tagging statistics

### Input File Formats

#### Text File (.txt)
- One item description per line
- Empty lines are ignored
- Example:
```text
11737 5/8-4-8 TYPE X DRYWALL
42418 PURDY 2.5-IN XL GLIDE ANG
ITEM # 476043 TAPCON 1/4-IN X 7-IN SDS DRILLBIT
```

#### CSV File (.csv)
- First column should contain item descriptions
- Headers are automatically detected
- Empty rows are skipped
- Example:
```csv
Item Description,SKU,Price
11737 5/8-4-8 TYPE X DRYWALL,SKU123,$10.99
42418 PURDY 2.5-IN XL GLIDE ANG,SKU456,$15.99
```

### Output Format
The script generates output in Python list format:
```python
labeled_data = [
    {
     "tokens": ["11737", "5/8", "-", "4", "-", "8", "TYPE", "X", "DRYWALL"],
     "ner_tags": ["B-ITEMNUM", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE"]
    },
    ...
]
```

### Tag Categories
The script recognizes the following entity types:

1. **ITEMNUM**: Item/product numbers
   - Example: "11737", "42418"

2. **BRAND**: Manufacturing brands
   - Example: "PURDY", "TAPCON", "DEWALT"

3. **SIZE**: Measurements and dimensions
   - Example: "5/8", "2.5-IN", "10-FT"

4. **SPEC**: Product specifications
   - Example: "TYPE X", "HEAVY DUTY", "FAST DRY"

5. **TYPE**: Product types
   - Example: "DRYWALL", "NAIL", "BRUSH"

6. **QUANTITY**: Amount or volume
   - Example: "100 CT", "5 LB", "4.5-GAL"

7. **PACKAGING**: Packaging types
   - Example: "BOX", "PAIL", "CASE"

### Examples

1. Basic processing:
```bash
python ner_tagger.py items.txt
```

2. Save output to file:
```bash
python ner_tagger.py items.csv -o tagged_items.py
```

3. Process and show statistics:
```bash
python ner_tagger.py items.txt --stats > output.py
```

### Statistics Output
When using the `--stats` flag, the script provides:
- Total number of items processed
- Count and percentage for each tag type
- Example output:
```
Read 150 items from items.txt

Tag Statistics:
B-ITEMNUM     150  (10.5%)
B-BRAND       145   (9.8%)
B-SIZE        200  (13.5%)
I-SIZE        300  (20.3%)
B-SPEC        120   (8.1%)
...
```

### Error Handling
The script includes error handling for common issues:

1. File not found:
```
Error reading file: [Errno 2] No such file or directory: 'missing.txt'
```

2. Invalid file format:
```
Error reading file: Invalid CSV format in line 5
```

3. Output file writing error:
```
Error saving to file: Permission denied
Printing results to stdout instead:
```

### Best Practices

1. **Input Data Preparation**:
   - Clean up inconsistent formatting
   - Remove any special characters that aren't part of the item description
   - Ensure consistent spacing between words

2. **File Organization**:
   - Use descriptive file names
   - Keep input files in a dedicated directory
   - Use appropriate file extensions (.txt or .csv)

3. **Output Management**:
   - Use meaningful output file names
   - Include date or version in output files if needed
   - Review statistics to verify tagging quality

### Known Limitations

1. Token Recognition:
   - Complex product codes might be split incorrectly
   - Very long descriptions might have inconsistent tagging
   - Uncommon unit measurements might not be recognized

2. Brand Recognition:
   - Limited to pre-defined brand list
   - New or uncommon brands might be missed
   - Brand variations might not be recognized

3. Multi-word Entities:
   - Complex multi-word brands might be split
   - Unusual product names might not be properly tagged
   - Compound measurements might be inconsistently tagged

### Troubleshooting

1. **Incorrect Tagging**:
   - Check if the entity is in the known lists (BRANDS, TYPES, etc.)
   - Verify the input format matches expected patterns
   - Review similar correctly tagged items for comparison

2. **File Processing Issues**:
   - Verify file encoding (should be UTF-8)
   - Check for invalid characters in the input
   - Ensure proper file permissions

3. **Performance Issues**:
   - Consider splitting very large files
   - Remove unnecessary empty lines
   - Clean up input data before processing

### Contact & Support
For issues, suggestions, or contributions:
- Report issues with example input and expected output
- Include error messages and stack traces if applicable
- Provide system information (Python version, OS)