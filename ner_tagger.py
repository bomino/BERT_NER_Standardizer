
"""
Enhanced NER Tagger for Item Descriptions
Part 1: Base Configuration and Imports
"""

import re
import csv
import json
import yaml
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any

# Custom exceptions
class NERTaggerError(Exception):
    """Base exception for NER tagger errors"""
    pass

class ValidationError(NERTaggerError):
    """Raised when input validation fails"""
    pass

class TokenizationError(NERTaggerError):
    """Raised when tokenization fails"""
    pass

class TaggingError(NERTaggerError):
    """Raised when tagging process fails"""
    pass

class ConfigurationError(NERTaggerError):
    """Raised when configuration is invalid"""
    pass

class Config:
    """Configuration management for the NER tagger"""
    def __init__(self, config_file: Optional[str] = None):
        self.config = {
            'max_item_length': 1000,
            'min_token_length': 1,
            'ignore_tokens': {',', ';', '.', '"', "'", '[', ']', '(', ')', '{', '}'},
            'ignore_words': {'and', 'with', 'for', 'the', 'a', 'an', 'of', 'to', 'in', 'at', 'by', 'up', 'w'},
            
            # Measurement unit aliases
            'unit_aliases': {
                # Length/Distance
                'INCH': 'IN', 'INCHES': 'IN',
                'FEET': 'FT', 'FOOT': 'FT',
                'CENTIMETER': 'CM', 'CENTIMETERS': 'CM',
                'MILLIMETER': 'MM', 'MILLIMETERS': 'MM',
                'METER': 'M', 'METERS': 'M',
                'YARD': 'YD', 'YARDS': 'YD',
                
                # Volume
                'GALLON': 'GAL', 'GALLONS': 'GAL',
                'OUNCE': 'OZ', 'OUNCES': 'OZ',
                'FLUID': 'FL', 'FLOZ': 'FL OZ',
                'QUART': 'QT', 'QUARTS': 'QT',
                'PINT': 'PT', 'PINTS': 'PT',
                'LITER': 'L', 'LITERS': 'L',
                'MILLILITER': 'ML', 'MILLILITERS': 'ML',
                
                # Weight
                'POUND': 'LB', 'POUNDS': 'LB',
                'KILOGRAM': 'KG', 'KILOGRAMS': 'KG',
                'GRAM': 'G', 'GRAMS': 'G',
                
                # Packaging
                'BOX': 'BX', 'BOXES': 'BX',
                'PACKAGE': 'PKG', 'PACKAGES': 'PKG',
                'PACK': 'PK', 'PACKS': 'PK',
                'PIECE': 'PC', 'PIECES': 'PC',
                'PALLET': 'PLT', 'PALLETS': 'PLT',
                'CASE': 'CS', 'CASES': 'CS',
                'CARTON': 'CTN', 'CARTONS': 'CTN',
                'BUNDLE': 'BDL', 'BUNDLES': 'BDL',
                'ROLL': 'RL', 'ROLLS': 'RL',
                'CONTAINER': 'CNTR', 'CONTAINERS': 'CNTR',
                'BUCKET': 'BKT', 'BUCKETS': 'BKT',
                'PAIL': 'PL', 'PAILS': 'PL',
                'BAG': 'BG', 'BAGS': 'BG',
                'TUBE': 'TB', 'TUBES': 'TB',
                'DRUM': 'DRM', 'DRUMS': 'DRM',
                'BOTTLE': 'BTL', 'BOTTLES': 'BTL',
                'EACH': 'EA',
                'COUNT': 'CT', 'COUNTS': 'CT',
                'SET': 'ST', 'SETS': 'ST',
                
                # Construction measurements
                'GAUGE': 'GA',
                'CALIBER': 'CAL',
                'DIAMETER': 'DIA',
                'SQUARE': 'SQ',
                'CUBIC': 'CU',
                'LINEAR': 'LN',
                'WIDTH': 'W',
                'HEIGHT': 'H',
                'DEPTH': 'D',
                'LENGTH': 'L',
                'THICKNESS': 'THK'
            },
            
            # Compound units
            'compound_units': {
                'SQUARE FEET': 'SQ FT',
                'SQUARE FOOT': 'SQ FT',
                'CUBIC FEET': 'CU FT',
                'CUBIC FOOT': 'CU FT',
                'LINEAR FEET': 'LN FT',
                'LINEAR FOOT': 'LN FT',
                'BOARD FEET': 'BF',
                'BOARD FOOT': 'BF',
                'SQUARE YARD': 'SQ YD',
                'SQUARE YARDS': 'SQ YD',
                'CUBIC YARD': 'CU YD',
                'CUBIC YARDS': 'CU YD'
            }
        }
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str) -> None:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    custom_config = json.load(f)
                elif config_file.endswith(('.yaml', '.yml')):
                    custom_config = yaml.safe_load(f)
                else:
                    raise ConfigurationError("Unsupported config file format")
                self._update_config(custom_config)
        except Exception as e:
            logging.warning(f"Error loading config file: {e}")
            logging.warning("Using default configuration")

    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                logging.warning(f"Unknown configuration key: {key}")

    def get_unit_alias(self, unit: str) -> str:
        """Get standardized unit from any variant"""
        unit_upper = unit.upper()
        return self.config['unit_aliases'].get(unit_upper, unit_upper)

    def get_compound_unit(self, unit_phrase: str) -> str:
        """Get standardized compound unit from phrase"""
        unit_upper = unit_phrase.upper()
        return self.config['compound_units'].get(unit_upper, unit_upper)

    def is_known_unit(self, unit: str) -> bool:
        """Check if a unit is in our known units list"""
        unit_upper = unit.upper()
        return (unit_upper in self.config['unit_aliases'].values() or
                unit_upper in self.config['unit_aliases'].keys() or
                unit_upper in self.config['compound_units'].values() or
                unit_upper in self.config['compound_units'].keys())

    def should_ignore_token(self, token: str) -> bool:
        """Check if token should be ignored"""
        return (token in self.config['ignore_tokens'] or
                token.lower() in self.config['ignore_words'])



"""
Enhanced NER Tagger for Item Descriptions
Part 2: Entity Detection and Tokenization
"""

class EntityDetector:
    """Detects and classifies entities in tokens"""
    def __init__(self):
        self.config = Config()
        self.load_entity_lists()

    def load_entity_lists(self) -> None:
        """Load entity lists for classification"""
        self.BRANDS = {
            # Common construction/hardware brands
            'PURDY', 'USG', 'EZ', 'ANCOR', 'TAPCON', 'DEWALT', 'QUIKRETE',
            'GORILLA', 'MINWAX', 'RAMBOARD', 'ZINSSER', 'QUICKIE', 'BERCOM',
            'BONDO', 'TITEBOND', 'ROCKWOOL', 'HARDIE', 'DYNAFLEX', 'GTR',
            'WHIZZ', 'PROJECT', 'SOURCE', 'IRWIN', 'SCOTCH', 'BLUE', 'TEKS',
            'BISSELL', 'CERTAINTEED', 'CLARK', 'DIETRICH', 'NOCOAT', 'ULTRAFLEX',
            'GREAT', 'STUFF', 'NELSON', 'ROMAN', 'PFISTER', 'MOEN', 'DELTA',
            'KOHLER', 'SIMPSON', 'STRONGTIE', 'RUSTOLEUM', 'KILZ', 'BEHR',
            'MAKITA', 'BOSCH', 'RYOBI', 'MILWAUKEE', 'RIDGID', 'HUSKY',
            'WERNER', 'LOUISVILLE', 'KLEIN', 'CHANNELLOCK', 'CRAFTSMAN',
            # Add more brands as needed
        }

        self.TYPES = {
            # Building materials
            'DRYWALL', 'SHEETROCK', 'PLYWOOD', 'LUMBER', 'BOARD', 'PANEL',
            'INSULATION', 'CONCRETE', 'CEMENT', 'MORTAR', 'GROUT', 'CAULK',
            'ADHESIVE', 'SEALANT',
            
            # Hardware and fasteners
            'NAIL', 'SCREW', 'BOLT', 'NUT', 'WASHER', 'ANCHOR', 'STAPLE',
            'RIVET', 'PIN', 'CLIP', 'BRACKET', 'HINGE', 'LATCH', 'LOCK',
            
            # Tools and accessories
            'DRILL', 'SAW', 'HAMMER', 'WRENCH', 'PLIERS', 'SCREWDRIVER',
            'BLADE', 'BIT', 'ROLLER', 'BRUSH', 'TAPE', 'KNIFE', 'CUTTER',
            'MEASURE', 'LEVEL', 'SQUARE', 'CHISEL', 'FILE', 'CLAMP',
            
            # Plumbing
            'PIPE', 'FITTING', 'VALVE', 'COUPLING', 'ELBOW', 'TEE', 'CAP',
            'ADAPTER', 'FLANGE', 'GASKET', 'TRAP', 'DRAIN', 'STRAINER',
            
            # Electrical
            'WIRE', 'CABLE', 'CONDUIT', 'OUTLET', 'SWITCH', 'BOX', 'CONNECTOR',
            'BREAKER', 'FUSE', 'TIMER', 'SENSOR',
            
            # Structural
            'STUD', 'BEAM', 'JOIST', 'BRACE', 'PLATE', 'CHANNEL', 'TRACK',
            'ANGLE', 'STRAP', 'HANGER',
            
            # Finishes
            'PAINT', 'STAIN', 'VARNISH', 'PRIMER', 'SEALER', 'FINISH',
            'COATING', 'WAX', 'POLISH',
            
            # Misc
            'TOOL', 'MATERIAL', 'SUPPLY', 'EQUIPMENT', 'ACCESSORY', 'KIT',
            'SYSTEM', 'ASSEMBLY', 'COMPOUND', 'MIXTURE', 'SOLUTION'
        }

        self.SPECS = {
            # Grades and ratings
            'TYPE', 'GRADE', 'CLASS', 'SERIES', 'RATING', 'RATED', 'CERTIFIED',
            'APPROVED', 'LISTED', 'COMPLIANT',
            
            # Physical properties
            'HEAVY', 'LIGHT', 'THICK', 'THIN', 'WIDE', 'NARROW', 'DEEP',
            'SHALLOW', 'LONG', 'SHORT', 'LARGE', 'SMALL',
            
            # Quality and durability
            'DUTY', 'STRENGTH', 'RESISTANT', 'PROOF', 'GUARD', 'PROTECTED',
            'PREMIUM', 'PROFESSIONAL', 'COMMERCIAL', 'INDUSTRIAL', 'RESIDENTIAL',
            
            # Performance characteristics
            'FAST', 'QUICK', 'RAPID', 'INSTANT', 'SLOW', 'PERMANENT',
            'TEMPORARY', 'FLEXIBLE', 'RIGID', 'SOLID', 'HOLLOW',
            
            # Material properties
            'GALVANIZED', 'STAINLESS', 'COATED', 'TREATED', 'FINISHED',
            'PRIMED', 'PAINTED', 'BARE', 'RAW',
            
            # Standards and codes
            'STANDARD', 'CODE', 'SPEC', 'UL', 'ASTM', 'ANSI', 'CSA',
            
            # Color and appearance
            'WHITE', 'BLACK', 'GRAY', 'RED', 'BLUE', 'GREEN', 'BROWN',
            'CLEAR', 'TRANSPARENT', 'OPAQUE', 'GLOSS', 'MATTE', 'SATIN',
            
            # Common product specs
            'X', 'XL', 'HD', 'PRO', 'MAX', 'PLUS', 'ULTRA', 'SUPER',
            'PREMIUM', 'BASIC', 'STANDARD', 'CUSTOM', 'UNIVERSAL',
            
            # Construction-specific
            'CSJ', 'CSS', 'TSB', 'GA', 'GAUGE', 'GRIT', 'TPI', 'PSI',
            'UNFACED', 'FACED', 'KRAFT', 'FIRE', 'STRUCTURAL'
        }

    def is_item_number(self, token: str, index: int, tokens: List[str]) -> bool:
        """
        Check if token is an item/product number
        Args:
            token: Token to check
            index: Position of token in sequence
            tokens: Complete token sequence
        """
        # Early position check (item numbers usually appear at start)
        if index > 2 and token not in {'#', 'ITEM', 'NO', 'NUMBER'}:
            return False

        # Common item number patterns
        patterns = [
            r'^\d{5,8}$',                    # Standard numeric (5-8 digits)
            r'^\d{3,6}[A-Z]+\d*$',           # Number followed by letters
            r'^[A-Z]{2,4}\d{3,6}$',          # Letters followed by numbers
            r'^\d+[A-Z]+\d+$',               # Mixed alphanumeric
            r'^[A-Z]+\d+[A-Z]+$',            # Letters-numbers-letters
            r'^\d{2,4}-\d{2,4}$',            # Hyphenated numbers
            r'^[A-Z]{1,3}-\d{3,6}$'          # Letters-hyphen-numbers
        ]
        
        # Check if token matches any item number pattern
        if any(re.match(pattern, token) for pattern in patterns):
            return True
            
        # Check for item number context
        if index > 0 and tokens[index-1] in {'ITEM', '#', 'NO', 'NUMBER'}:
            return token.isalnum()
            
        return False

    def is_measurement(self, token: str) -> bool:
        """
        Check if token represents a measurement
        """
        # Measurement patterns
        patterns = [
            r'^\d+(/\d+)?(\.\d+)?$',         # Numbers including fractions
            r'^\d+\.\d+$',                    # Decimal numbers
            r'^\d+(/\d+)?([xX]\d+)*$',       # Dimensions
            r'^\d+(\.\d+)?(MM|CM|M|IN|FT|YD)$',  # Numbers with units
            r'^\d+(/\d+)?[xX]\d+(/\d+)?$'    # Fractional dimensions
        ]
        
        # Basic pattern check
        if any(re.match(pattern, token) for pattern in patterns):
            return True
            
        # Unit combinations check
        if re.match(r'^\d+(\.\d+)?(IN|FT|MM|CM|OZ|LB|GA|FL|GAL)$', token):
            return True
            
        # Fractional measurements
        if re.match(r'^\d+/\d+$', token):
            return True
            
        return False

    def is_brand(self, token: str, prev_token: Optional[str] = None) -> bool:
        """Check if token is part of a brand name"""
        if token in self.BRANDS:
            return True
        if prev_token and f"{prev_token} {token}" in self.BRANDS:
            return True
        return False

    def is_type(self, token: str, prev_token: Optional[str] = None) -> bool:
        """Check if token is part of a product type"""
        if token in self.TYPES:
            return True
        if prev_token and f"{prev_token} {token}" in self.TYPES:
            return True
        return False

    def is_spec(self, token: str, prev_token: Optional[str] = None) -> bool:
        """Check if token is part of a specification"""
        if token in self.SPECS:
            return True
        if prev_token and f"{prev_token} {token}" in self.SPECS:
            return True
        if token.endswith('GA') or token.endswith('GAUGE'):
            return True
        return False


class Tokenizer:
    """Handles tokenization of item descriptions"""
    def __init__(self, config: Config):
        self.config = config

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual components
        Args:
            text: Input text to tokenize
        Returns:
            List of tokens
        """
        # Pre-process text
        text = self._preprocess_text(text)
        
        tokens = []
        current_token = ''
        
        # Process text character by character
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.extend(self._handle_token(current_token))
                    current_token = ''
            elif char in self.config.config['ignore_tokens']:
                if current_token:
                    tokens.extend(self._handle_token(current_token))
                    current_token = ''
                tokens.append(char)
            elif char in {'-', 'X', 'x', '/', '×'}:
                if current_token:
                    tokens.extend(self._handle_token(current_token))
                    current_token = ''
                tokens.append(char.upper())
            else:
                current_token += char
        
        # Handle final token
        if current_token:
            tokens.extend(self._handle_token(current_token))
        
        # Post-process tokens
        return self._postprocess_tokens(tokens)

    def _preprocess_text(self, text: str) -> str:
        """Prepare text for tokenization"""
        # Standardize item number indicators
        text = text.replace('ITEM#', 'ITEM #')
        text = text.replace('ITEM NO.', 'ITEM #')
        text = text.replace('ITEM NO', 'ITEM #')
        text = text.replace('#', ' # ')
        
        # Split alphanumeric combinations
        text = re.sub(r'(\d+)([A-Z]+)', r'\1 \2', text)  # "35818CSJ" → "35818 CSJ"
        text = re.sub(r'([A-Z]+)(\d+)', r'\1 \2', text)  # "CSJ35818" → "CSJ 35818"
        
        # Standardize separators
        text = text.replace('×', 'X')  # Replace multiplication symbol with X
        text = text.replace('""', '"')  # Replace double quotes
        text = text.replace('``', '"')  # Replace backticks
        
        return text.upper()  # Convert to uppercase for consistency

    def _handle_token(self, token: str) -> List[str]:
        """Process individual token and handle special cases"""
        # Skip empty tokens
        if not token:
            return []
            
        # Handle measurements with units
        match = re.match(r'^(\d+(\.\d+)?)(IN|FT|MM|CM|OZ|LB|GA|FL|GAL)$', token)
        if match:
            return [match.group(1), match.group(3)]
            
        # Handle fractions
        if re.match(r'^\d+/\d+$', token):
            return [token]
            
        # Handle dimensions
        match = re.match(r'^(\d+)X(\d+)$', token)
        if match:
            return [match.group(1), 'X', match.group(2)]
            
        # Handle compound measurements
        match = re.match(r'^(\d+)-(\d+/\d+)$', token)
        if match:
            return [match.group(1), '-', match.group(2)]
            
        return [token]

    def _postprocess_tokens(self, tokens: List[str]) -> List[str]:
        """Clean up and standardize tokens"""
        cleaned_tokens = []
        skip_next = False
        
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
                
            # Skip ignored tokens
            if self.config.should_ignore_token(token):
                continue
                
            # Handle compound units
            if i < len(tokens) - 1:
                compound = f"{token} {tokens[i+1]}"
                if compound in self.config.config['compound_units']:
                    cleaned_tokens.append(self.config.get_compound_unit(compound))
                    skip_next = True
                    continue
            
            # Standardize units
            if self.config.is_known_unit(token):
                cleaned_tokens.append(self.config.get_unit_alias(token))
            else:
                cleaned_tokens.append(token)
        
        return [t.strip() for t in cleaned_tokens if t.strip() and len(t.strip()) >= self.config.config['min_token_length']]

    def _is_numeric_with_unit(self, token: str) -> bool:
        """Check if token is a number with a unit"""
        return bool(re.match(r'^\d+(\.\d+)?[A-Z]+$', token))

    def _split_numeric_with_unit(self, token: str) -> List[str]:
        """Split a number-unit combination into separate tokens"""
        match = re.match(r'^(\d+(\.\d+)?)([A-Z]+)$', token)
        if match:
            return [match.group(1), match.group(3)]
        return [token]

    def validate_tokens(self, tokens: List[str]) -> None:
        """
        Validate tokenization results
        Raises TokenizationError if validation fails
        """
        if not tokens:
            raise TokenizationError("Tokenization produced no tokens")
        
        for token in tokens:
            if not token.strip():
                raise TokenizationError(f"Empty token found in sequence: {tokens}")
            if len(token) < self.config.config['min_token_length']:
                raise TokenizationError(f"Token '{token}' is too short")
            if len(token) > self.config.config['max_item_length']:
                raise TokenizationError(f"Token '{token}' is too long")

class TokenizedItem:
    """
    Represents a tokenized item with additional metadata
    Used for tracking original text and token positions
    """
    def __init__(self, original_text: str, tokens: List[str]):
        self.original_text = original_text
        self.tokens = tokens
        self.token_spans = self._get_token_spans()
        
    def _get_token_spans(self) -> List[Tuple[int, int]]:
        """Calculate the start and end positions of each token in original text"""
        spans = []
        start = 0
        for token in self.tokens:
            # Find token in original text starting from last end position
            while start < len(self.original_text):
                pos = self.original_text.upper().find(token, start)
                if pos != -1:
                    spans.append((pos, pos + len(token)))
                    start = pos + len(token)
                    break
                start += 1
        return spans
    
    def get_token_context(self, index: int, window: int = 2) -> Dict[str, Optional[str]]:
        """Get surrounding context for a token"""
        return {
            'prev2': self.tokens[index - 2] if index >= 2 else None,
            'prev1': self.tokens[index - 1] if index >= 1 else None,
            'next1': self.tokens[index + 1] if index < len(self.tokens) - 1 else None,
            'next2': self.tokens[index + 2] if index < len(self.tokens) - 2 else None
        }       



"""
Enhanced NER Tagger for Item Descriptions
Part 3: Tagger Implementation
"""

class Tagger:
    """Assigns BIO tags to tokens"""
    def __init__(self, entity_detector: EntityDetector):
        self.detector = entity_detector
        self._tag_cache = {}
        self.tag_sequence_rules = {
            'B-BRAND': {'B-TYPE', 'B-SPEC', 'B-SIZE', 'I-BRAND'},
            'B-TYPE': {'B-SPEC', 'B-SIZE', 'B-QUANTITY', 'I-TYPE'},
            'B-SPEC': {'B-TYPE', 'B-SIZE', 'I-SPEC'},
            'B-SIZE': {'B-TYPE', 'B-SPEC', 'I-SIZE'},
            'B-QUANTITY': {'B-TYPE', 'B-PACKAGING', 'I-QUANTITY'},
            'B-PACKAGING': {'O', 'B-BRAND', 'B-TYPE'}
        }

    def tag_tokens(self, tokens: List[str]) -> List[str]:
        """
        Generate BIO tags for a list of tokens
        Args:
            tokens: List of tokens to tag
        Returns:
            List of BIO tags corresponding to tokens
        """
        tags = []
        for i, token in enumerate(tokens):
            tag = self._get_token_tag(token, i, tokens)
            # Apply sequence rules
            if i > 0 and not self._is_valid_sequence(tags[-1], tag):
                tag = self._adjust_tag(tag, tags[-1])
            tags.append(tag)
        return tags

    def _get_token_tag(self, token: str, index: int, tokens: List[str]) -> str:
        """
        Get the BIO tag for a single token with context
        """
        # Check cache
        cache_key = (token, index, tuple(tokens))
        if cache_key in self._tag_cache:
            return self._tag_cache[cache_key]

        # Get context
        context = self._get_context(index, tokens)
        
        # Determine tag
        tag = self._determine_tag(token, index, tokens, context)
        
        # Cache and return
        self._tag_cache[cache_key] = tag
        return tag

    def _get_context(self, index: int, tokens: List[str]) -> Dict[str, Optional[str]]:
        """Get context information for a token"""
        return {
            'prev_token': tokens[index - 1] if index > 0 else None,
            'next_token': tokens[index + 1] if index < len(tokens) - 1 else None,
            'prev_prev_token': tokens[index - 2] if index > 1 else None,
            'next_next_token': tokens[index + 2] if index < len(tokens) - 2 else None
        }

    def _determine_tag(self, token: str, index: int, tokens: List[str], 
                      context: Dict[str, Optional[str]]) -> str:
        """
        Determine the appropriate BIO tag for a token based on rules and context
        """
        # Check for item numbers
        if self.detector.is_item_number(token, index, tokens):
            return 'B-ITEMNUM'

        # Get the previous tag if it exists
        prev_tag = self._get_token_tag(tokens[index-1], index-1, tokens) if index > 0 else None

        # Check for continuation of previous entity
        if prev_tag:
            continued_tag = self._check_continuation(token, prev_tag, context)
            if continued_tag:
                return continued_tag

        # Check for measurements/sizes
        if self.detector.is_measurement(token):
            return 'B-SIZE'

        # Check for brands
        if self.detector.is_brand(token, context.get('prev_token')):
            return 'B-BRAND'

        # Check for types
        if self.detector.is_type(token, context.get('prev_token')):
            return 'B-TYPE'

        # Check for specifications
        if self.detector.is_spec(token, context.get('prev_token')):
            return 'B-SPEC'

        # Check for quantities
        if self._is_quantity(token, context):
            return 'B-QUANTITY'

        return 'O'

    def _check_continuation(self, token: str, prev_tag: str, 
                          context: Dict[str, Optional[str]]) -> Optional[str]:
        """Check if token continues the previous entity"""
        if not prev_tag.startswith(('B-', 'I-')):
            return None

        entity_type = prev_tag[2:]  # Remove B- or I- prefix
        
        # Size continuation
        if entity_type == 'SIZE':
            if token in {'-', 'X', '/'} or self.detector.is_measurement(token):
                return f'I-{entity_type}'
                
        # Spec continuation
        elif entity_type == 'SPEC':
            if self.detector.is_spec(token, context.get('prev_token')):
                return f'I-{entity_type}'
                
        # Brand continuation
        elif entity_type == 'BRAND':
            if self.detector.is_brand(token, context.get('prev_token')):
                return f'I-{entity_type}'
                
        # Type continuation
        elif entity_type == 'TYPE':
            if self.detector.is_type(token, context.get('prev_token')):
                return f'I-{entity_type}'
                
        # Quantity continuation
        elif entity_type == 'QUANTITY':
            if self._is_quantity_continuation(token, context):
                return f'I-{entity_type}'

        return None

    def _is_quantity(self, token: str, context: Dict[str, Optional[str]]) -> bool:
        """Check if token represents a quantity"""
        if token.isdigit():
            next_token = context.get('next_token')
            if next_token in {'CT', 'PC', 'PCS', 'BOX', 'PKG', 'EA', 'SET'}:
                return True
        return False

    def _is_quantity_continuation(self, token: str, context: Dict[str, Optional[str]]) -> bool:
        """Check if token continues a quantity"""
        return token in {'CT', 'PC', 'PCS', 'BOX', 'PKG', 'EA', 'SET'} and \
               context.get('prev_token', '').isdigit()

    def _is_valid_sequence(self, prev_tag: str, current_tag: str) -> bool:
        """Check if tag sequence is valid according to rules"""
        if prev_tag.startswith('B-'):
            valid_next = self.tag_sequence_rules.get(prev_tag, set())
            return current_tag in valid_next
        return True

    def _adjust_tag(self, tag: str, prev_tag: str) -> str:
        """Adjust tag based on sequence rules"""
        if tag.startswith('B-') and prev_tag.startswith('B-'):
            if prev_tag[2:] == tag[2:]:
                return 'I-' + tag[2:]
        return tag     



"""
Enhanced NER Tagger for Item Descriptions
Part 4: Statistics, Processing, and Main Implementation
"""

class TaggingStatistics:
    """Collects and reports statistics about the tagging process"""
    def __init__(self):
        self.total_items = 0
        self.total_tokens = 0
        self.tag_counts = {}
        self.entity_lengths = {
            'BRAND': [],
            'TYPE': [],
            'SPEC': [],
            'SIZE': [],
            'QUANTITY': [],
            'ITEMNUM': []
        }
        self.sequence_patterns = {}
        self.token_patterns = {}
        self.errors = []

    def update(self, tokens: List[str], tags: List[str]) -> None:
        """Update statistics with new item data"""
        self.total_items += 1
        self.total_tokens += len(tokens)

        # Update tag counts
        for tag in tags:
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1

        # Track entity lengths
        current_entity = None
        current_length = 0
        
        for tag in tags:
            if tag.startswith('B-'):
                if current_entity:
                    self.entity_lengths[current_entity].append(current_length)
                current_entity = tag[2:]
                current_length = 1
            elif tag.startswith('I-'):
                current_length += 1
            elif current_entity:
                self.entity_lengths[current_entity].append(current_length)
                current_entity = None
                current_length = 0

        # Track sequence patterns
        sequence = []
        for tag in tags:
            if tag != 'O':
                sequence.append(tag)
            elif sequence:
                pattern = ' '.join(sequence)
                self.sequence_patterns[pattern] = self.sequence_patterns.get(pattern, 0) + 1
                sequence = []

        # Track token patterns
        for token, tag in zip(tokens, tags):
            if tag != 'O':
                self.token_patterns[token] = self.token_patterns.get(token, set())
                self.token_patterns[token].add(tag)

    def add_error(self, item: str, error: str) -> None:
        """Record processing errors"""
        self.errors.append((item, error))

    def print_report(self) -> None:
        """Print comprehensive statistical report"""
        print("\n=== Tagging Statistics Report ===")
        print(f"\nProcessed {self.total_items} items with {self.total_tokens} tokens")
        
        print("\nTag Distribution:")
        total_tags = sum(self.tag_counts.values())
        for tag, count in sorted(self.tag_counts.items()):
            percentage = (count / total_tags) * 100
            print(f"{tag:12} {count:5d} ({percentage:5.1f}%)")

        print("\nEntity Length Statistics:")
        for entity, lengths in self.entity_lengths.items():
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                max_length = max(lengths)
                print(f"{entity:12} Avg: {avg_length:4.1f} Max: {max_length:2d}")

        print("\nMost Common Sequence Patterns:")
        for pattern, count in sorted(self.sequence_patterns.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"{pattern:30} {count:5d}")

        if self.errors:
            print("\nProcessing Errors:")
            for item, error in self.errors:
                print(f"Item: {item[:50]}... Error: {error}")

class ItemProcessor:
    """Main class for processing items"""
    def __init__(self, config_file: Optional[str] = None):
        self.config = Config(config_file)
        self.entity_detector = EntityDetector()
        self.tokenizer = Tokenizer(self.config)
        self.tagger = Tagger(self.entity_detector)
        self.stats = TaggingStatistics()

    def _clean_text(self, text: str) -> str:
        """Clean text by replacing special characters with standard ASCII equivalents"""
        replacements = {
            '"': '"',    # Replace curly quotes with straight quotes
            '"': '"',
            ''': "'",    # Replace curly apostrophes with straight apostrophes
            ''': "'",
            '–': '-',    # Replace en-dash with hyphen
            '—': '-',    # Replace em-dash with hyphen
            '×': 'X',    # Replace multiplication symbol with X
            '½': '1/2',  # Replace common fractions
            '¼': '1/4',
            '¾': '3/4',
            '⅛': '1/8',
            '⅜': '3/8',
            '⅝': '5/8',
            '⅞': '7/8',
            # Common measurement symbols
            '"': 'IN',
            '″': 'IN',
            '′': 'FT',
            "'": 'FT',
            '°': 'DEG',
            'ø': 'DIA',
            '®': '',     # Remove registered trademark
            '™': '',     # Remove trademark
            '©': '',     # Remove copyright
            # Add any other special characters as needed
        }
        
        # Apply replacements
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Additional cleaning steps
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # normalize spaces
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def process_items(self, items: List[str]) -> List[Dict]:
        """Process a list of items"""
        try:
            self._validate_items(items)
            results = []
            
            for item in items:
                try:
                    # Process single item
                    result = self.process_single_item(item)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.stats.add_error(item, str(e))
                    logging.warning(f"Error processing item '{item[:50]}...': {e}")
            
            return results
            
        except ValidationError as e:
            logging.error(f"Validation error: {e}")
            return []

    def process_single_item(self, item: str) -> Optional[Dict]:
        """Process a single item description"""
        try:
            # Clean the text first
            cleaned_item = self._clean_text(item)
            
            # Tokenize
            tokens = self.tokenizer.tokenize(cleaned_item)
            self.tokenizer.validate_tokens(tokens)
            
            # Generate tags
            tags = self.tagger.tag_tokens(tokens)
            
            # Update statistics
            self.stats.update(tokens, tags)
            
            return {
                "tokens": tokens,
                "ner_tags": tags
            }
            
        except (TokenizationError, TaggingError) as e:
            self.stats.add_error(item, str(e))
            return None

    def _validate_items(self, items: List[str]) -> None:
        """Validate input items"""
        if not items:
            raise ValidationError("No items to process")
            
        for item in items:
            if not isinstance(item, str):
                raise ValidationError(f"Invalid item type: {type(item)}")
            if not item.strip():
                raise ValidationError("Empty item description")
            if len(item) > self.config.config['max_item_length']:
                raise ValidationError(f"Item description too long: {item[:50]}...")
            
            # Clean the text instead of rejecting non-ASCII characters
            cleaned_item = self._clean_text(item)
            if not cleaned_item:
                raise ValidationError(f"Item becomes empty after cleaning: {item[:50]}...")

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ner_tagger.log')
        ]
    )

def read_input_file(file_path: str) -> List[str]:
    """Read items from input file"""
    items = []
    path = Path(file_path)
    
    try:
        if path.suffix.lower() == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                # Skip header if exists
                try:
                    header = next(csv_reader)
                    # If first row doesn't look like a header, add it back
                    if any(cell.strip().isdigit() for cell in header):
                        items.append(header[0])
                except StopIteration:
                    pass
                
                for row in csv_reader:
                    if row:  # Skip empty rows
                        items.append(row[0])
        else:  # Treat as text file
            with open(path, 'r', encoding='utf-8') as f:
                items = [line.strip() for line in f if line.strip()]
                
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        sys.exit(1)
        
    return items

def save_output(labeled_data: List[Dict], output_file: Optional[str] = None) -> None:
    """Save labeled data to file or print to stdout"""
    output = "labeled_data = [\n"
    for item in labeled_data:
        output += "    {\n"
        output += f"     \"tokens\": {item['tokens']},\n"
        output += f"     \"ner_tags\": {item['ner_tags']}"
        if item != labeled_data[-1]:
            output += "\n    },\n"
        else:
            output += "\n    }\n"
    output += "]"
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            logging.info(f"Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving to file: {e}")
            logging.info("Printing results to stdout instead:")
            print(output)
    else:
        print(output)


def save_output_csv(labeled_data: List[Dict], output_file: str) -> None:
    """
    Save labeled data to CSV file
    
    Args:
        labeled_data: List of dictionaries containing tokens and tags
        output_file: Path to output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Original_Text', 'Token', 'Tag'])
            
            # Write data
            for item in labeled_data:
                tokens = item['tokens']
                tags = item['ner_tags']
                original_text = ' '.join(tokens)
                
                # Write each token-tag pair
                for token, tag in zip(tokens, tags):
                    writer.writerow([original_text, token, tag])
                
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving to CSV file: {e}")
        raise        




def main():
    """Main function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process item descriptions and generate BIO tags.'
    )
    parser.add_argument('input_file', help='Path to input file (txt or csv)')
    parser.add_argument('-o', '--output', help='Path to output file (optional)')
    parser.add_argument('--format', choices=['py', 'csv'], default='py',
                       help='Output format (py for Python list or csv for CSV file)')
    parser.add_argument('--config', help='Path to configuration file (optional)')
    parser.add_argument('--stats', action='store_true', help='Show tagging statistics')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        # Initialize processor
        processor = ItemProcessor(args.config)

        # Read input items
        items = read_input_file(args.input_file)
        logging.info(f"Read {len(items)} items from {args.input_file}")

        # Process items
        labeled_data = processor.process_items(items)

        # Save or print results
        if args.output:
            if args.format == 'csv':
                save_output_csv(labeled_data, args.output)
            else:
                save_output(labeled_data, args.output)
        else:
            # Print to stdout in original format
            save_output(labeled_data, None)

        # Show statistics if requested
        if args.stats:
            processor.stats.print_report()

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()