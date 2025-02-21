from transformers import pipeline
import pandas as pd
import re

class MLConstructionStandardizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load fine-tuned BERT NER model
        self.ner_pipeline = pipeline("ner", model="bert_construction_ner", 
                                   tokenizer="bert_construction_ner", 
                                   aggregation_strategy="simple")
        self._initialize_patterns()

    def _initialize_patterns(self):
        # Your existing patterns (simplified for brevity)
        self.measurement_patterns = {
            r'(\d+)\s*/\s*(\d+)': r'\1/\2',
            r'(\d+\.?\d*).*(IN|INCH|INCHES)': r'\1IN',
        }
        self.unit_standardization = {
            r'\bINCHES\b|\bINCH\b|\bIN\b': 'IN',
            r'\bPOUNDS\b|\bLBS\b': 'LB',
        }
        self.construction_terms = {
            r'\bDRYWALL\b': 'DRYWALL',
            r'\bSCREW(S)?\b': 'SCR',
        }

    def standardize_description(self, text: str) -> str:
        if not text or pd.isna(text):
            return ''
        
        # Use BERT NER to extract components
        entities = self.ner_pipeline(text)
        components = {"BRAND": [], "TYPE": [], "SIZE": [], "SPEC": [], "QUANTITY": [], "ITEMNUM": [], "PACKAGING": []}
        for entity in entities:
            label = entity["entity_group"].split('-')[1]  # Extract base label (e.g., "B-BRAND" -> "BRAND")
            word = entity["word"].replace("##", "")  # Merge subwords
            components[label].append(word)
        
        # Merge multi-token entities
        for key in components:
            components[key] = " ".join(components[key]) if components[key] else ""
        
        # Fallback to regex if NER fails
        if not any(components.values()):
            standardized = text.upper()
            for pattern, repl in self.measurement_patterns.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            for pattern, repl in self.unit_standardization.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            for pattern, repl in self.construction_terms.items():
                standardized = re.sub(pattern, repl, standardized, flags=re.IGNORECASE)
            return re.sub(r'[^\w\s/.\-X]', '', standardized).strip()
        
        # Construct standardized description
        parts = [components[k] for k in ["BRAND", "TYPE", "SIZE", "SPEC", "QUANTITY"] if components[k]]
        return " ".join(parts)

    # Add your existing methods (learn_patterns, process_dataframe, etc.) here

# Test integration
standardizer = MLConstructionStandardizer()
test_desc = "ITEM # 476043 TAPCON 1/4-IN X 7-IN SDS DRILLBIT"
standardized = standardizer.standardize_description(test_desc)
print(f"Original: {test_desc}")
print(f"Standardized: {standardized}")