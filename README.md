# Construction Product Description Standardizer

This repository contains a Python-based solution for standardizing construction product descriptions using a fine-tuned BERT Named Entity Recognition (NER) model integrated with rule-based techniques. The project aims to extract and standardize key entities (e.g., `BRAND`, `TYPE`, `SIZE`, `SPEC`, `QUANTITY`, `ITEMNUM`, `PACKAGING`) from raw descriptions, ensuring consistency and usability in construction-related applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Training the Model](#training-the-model)
- [Performance](#performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The `MLConstructionStandardizer` class leverages a BERT model fine-tuned for NER to identify entities in construction product descriptions. It enhances accuracy with selective regex fallbacks, entity-specific standardization, and domain-specific term mappings. The solution evolved through iterative improvements addressing subword tokenization, label imbalance, and dataset size, culminating in a robust standardization tool.

## Features

- **BERT NER Integration**: Extracts entities using a fine-tuned `bert-base-uncased` model.
- **Subword Merging**: Combines BERT subwords (e.g., "47", "##60", "##43" → "476043") for cohesive entity output.
- **Selective Regex Fallback**: Applies regex to extract missed entities like `SIZE` when the model underperforms.
- **Entity Standardization**: Normalizes entities (e.g., "1/4-IN X 7-IN" → "1/4INX7IN") for consistency.
- **Domain-Specific Mapping**: Standardizes terms (e.g., "SCREW" → "SCR") using a predefined mapping.
- **Error Handling**: Includes logging and exception handling for robustness.

## Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (optional but recommended)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/bomino/BERT_NER_Standardizer.git
   cd construction-standardizer