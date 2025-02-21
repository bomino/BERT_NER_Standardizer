import logging
import re
import pandas as pd
from transformers import pipeline

class MLConstructionStandardizer:
    def __init__(self):
        """Initialize the standardizer with logging, NER pipeline, and predefined patterns."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load fine-tuned BERT NER model with confidence thresholding
        # Note: Replace "bert_construction_ner" with your actual model path if different
        self.ner_pipeline = pipeline("ner", model="bert_construction_ner", 
                                     tokenizer="bert_construction_ner", 
                                     aggregation_strategy="simple")
        self.confidence_threshold = 0.9  # Threshold for accepting NER predictions
        self._initialize_patterns()
        self._initialize_term_mapping()

    def _initialize_patterns(self):
        """Define regex patterns for entity extraction and standardization."""
        # Patterns for measurement standardization
        self.measurement_patterns = {
            r'(\d+)\s*/\s*(\d+)': r'\1/\2',  # e.g., "1 / 2" -> "1/2"
            r'(\d+\.?\d*).*(IN|INCH|INCHES)': r'\1IN',  # e.g., "2.5 INCHES" -> "2.5IN"
        }
        # Unit standardization
        self.unit_standardization = {
            r'\bINCHES\b|\bINCH\b|\bIN\b': 'IN',
            r'\bPOUNDS\b|\bLBS\b': 'LB',
        }
        # Construction-specific terms
        self.construction_terms = {
            r'\bDRYWALL\b': 'DRYWALL',
            r'\bSCREW(S)?\b': 'SCR',
        }
        # Pattern for size extraction
        self.size_pattern = r'(\d+/\d+|\d+\.?\d*)\s*(IN|INCH|INCHES)'

    def _initialize_term_mapping(self):
        """Define domain-specific term mappings for standardization."""
        self.term_mapping = {
            "SCREW": "SCR",
            "DRYWALL SCREW": "DRYWALL SCR",
            "SDS PLUS DRILLBIT": "SDS DRILLBIT",
        }

    def _standardize_size(self, size: str) -> str:
        """Standardize size format (e.g., '1/4-IN X 7-IN' -> '1/4INX7IN')."""
        size = re.sub(r'\s*INCHES?\b', 'IN', size, flags=re.IGNORECASE)
        size = re.sub(r'\s*X\s*', 'X', size)
        return size.upper()

    def _apply_term_mapping(self, text: str) -> str:
        """Apply domain-specific term standardization."""
        for term, replacement in self.term_mapping.items():
            text = re.sub(r'\b' + re.escape(term) + r'\b', replacement, text, flags=re.IGNORECASE)
        return text

    def _extract_with_regex(self, text: str, entity_type: str) -> str:
        """Extract entities using regex as a fallback if not detected by the model."""
        if entity_type == "SIZE":
            match = re.search(self.size_pattern, text, re.IGNORECASE)
            if match:
                return self._standardize_size(match.group(0))
        return ""

    def standardize_description(self, text: str) -> str:
        """
        Standardize a construction product description using NER and regex fallbacks.
        
        Args:
            text (str): The raw description to standardize.
        
        Returns:
            str: The standardized description.
        """
        # Handle empty or invalid input
        if not text or pd.isna(text):
            return ''
        
        # Attempt NER extraction with error handling
        try:
            entities = self.ner_pipeline(text)
            filtered_entities = [e for e in entities if e["score"] >= self.confidence_threshold]
            self.logger.info(f"Extracted entities: {filtered_entities}")
        except Exception as e:
            self.logger.error(f"NER pipeline failed for input '{text}': {str(e)}")
            filtered_entities = []

        # Initialize entity components
        components = {
            "BRAND": [], "TYPE": [], "SIZE": [], "SPEC": [], 
            "QUANTITY": [], "ITEMNUM": [], "PACKAGING": []
        }
        
        # Group entities by type
        for entity in filtered_entities:
            label = entity["entity_group"].split('-')[1]  # e.g., "B-BRAND" -> "BRAND"
            word = entity["word"].replace("##", "")  # Merge BERT subwords
            components[label].append(word)
        
        # Merge multi-token entities into single strings
        for key in components:
            components[key] = " ".join(components[key]) if components[key] else ""
        
        # Apply regex fallback for missing SIZE entity
        if not components["SIZE"]:
            components["SIZE"] = self._extract_with_regex(text, "SIZE")
            if components["SIZE"]:
                self.logger.info(f"Applied regex fallback for SIZE: {components['SIZE']}")
        
        # Standardize size if present
        if components["SIZE"]:
            components["SIZE"] = self._standardize_size(components["SIZE"])
        
        # Full regex fallback if no entities are detected
        if not any(components.values()):
            self.logger.info("No entities detected, applying full regex fallback")
            standardized = text.upper()
            for pattern, repl in self.measurement_patterns.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            for pattern, repl in self.unit_standardization.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            for pattern, repl in self.construction_terms.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            standardized = self._apply_term_mapping(standardized)
            return re.sub(r'[^\w\s/.\-X]', '', standardized).strip()
        
        # Construct standardized description with separators
        parts = [components[k] for k in ["BRAND", "TYPE", "SIZE", "SPEC", "QUANTITY"] if components[k]]
        standardized = " - ".join(parts)
        
        # Apply term mapping for consistency
        standardized = self._apply_term_mapping(standardized)
        
        return standardized

# Test the class
if __name__ == "__main__":
    standardizer = MLConstructionStandardizer()
    test_desc = "ITEM # 476043 TAPCON 1/4-IN X 7-IN SDS DRILLBIT"
    standardized = standardizer.standardize_description(test_desc)
    print(f"Original: {test_desc}")
    print(f"Standardized: {standardized}")