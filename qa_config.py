"""
Quality Assurance Configuration and Validation Utilities
For Insurance AI System
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None

    def __str__(self):
        icon = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠"}[self.status.value]
        return f"[{icon}] {self.field or 'Check'}: {self.message}"


# =============================================================================
# PDF/Document QA Configuration
# =============================================================================

class DocumentQA:
    """Quality Assurance for document processing"""

    # File validation
    REQUIRED_FILE_EXTENSIONS = ['.pdf']
    MIN_FILE_SIZE_KB = 10
    MAX_FILE_SIZE_KB = 50000

    # Content validation
    MIN_PAGES = 1
    MAX_PAGES = 500
    MIN_TEXT_LENGTH_PER_PAGE = 50

    # Chunk validation
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 1000
    MIN_CHUNK_OVERLAP = 10
    MAX_CHUNK_OVERLAP = 200
    MIN_CHUNKS = 1
    MAX_CHUNKS = 10000

    # Quality thresholds
    MIN_AVG_CHUNK_LENGTH = 200
    MAX_EMPTY_CHUNKS_PERCENT = 5.0

    @staticmethod
    def validate_file_path(path: str) -> ValidationResult:
        if not path:
            return ValidationResult(ValidationStatus.FAIL, "File path is empty", "file_path")
        if not any(path.lower().endswith(ext) for ext in DocumentQA.REQUIRED_FILE_EXTENSIONS):
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Invalid extension. Expected one of {DocumentQA.REQUIRED_FILE_EXTENSIONS}",
                "file_path",
                path
            )
        return ValidationResult(ValidationStatus.PASS, "File path is valid", "file_path")

    @staticmethod
    def validate_chunk_size(chunk_size: int) -> ValidationResult:
        if not DocumentQA.MIN_CHUNK_SIZE <= chunk_size <= DocumentQA.MAX_CHUNK_SIZE:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Chunk size {chunk_size} out of range [{DocumentQA.MIN_CHUNK_SIZE}, {DocumentQA.MAX_CHUNK_SIZE}]",
                "chunk_size",
                chunk_size
            )
        return ValidationResult(ValidationStatus.PASS, "Chunk size is valid", "chunk_size")

    @staticmethod
    def validate_chunk_overlap(overlap: int, chunk_size: int) -> ValidationResult:
        if overlap < 0:
            return ValidationResult(ValidationStatus.FAIL, "Overlap cannot be negative", "chunk_overlap", overlap)
        if overlap >= chunk_size:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Overlap ({overlap}) must be less than chunk size ({chunk_size})",
                "chunk_overlap",
                overlap
            )
        return ValidationResult(ValidationStatus.PASS, "Chunk overlap is valid", "chunk_overlap")

    @staticmethod
    def validate_chunks(chunks: List) -> List[ValidationResult]:
        results = []

        if len(chunks) < DocumentQA.MIN_CHUNKS:
            results.append(ValidationResult(
                ValidationStatus.FAIL,
                f"No chunks created. Expected at least {DocumentQA.MIN_CHUNKS}",
                "chunks",
                len(chunks)
            ))
            return results

        if len(chunks) > DocumentQA.MAX_CHUNKS:
            results.append(ValidationResult(
                ValidationStatus.WARNING,
                f"Large number of chunks ({len(chunks)}). May impact performance.",
                "chunks",
                len(chunks)
            ))

        empty_chunks = sum(1 for c in chunks if not c.page_content or len(c.page_content.strip()) == 0)
        empty_percent = (empty_chunks / len(chunks)) * 100

        if empty_percent > DocumentQA.MAX_EMPTY_CHUNKS_PERCENT:
            results.append(ValidationResult(
                ValidationStatus.FAIL,
                f"Too many empty chunks ({empty_percent:.1f}%). Max allowed: {DocumentQA.MAX_EMPTY_CHUNKS_PERCENT}%",
                "chunks",
                f"{empty_chunks}/{len(chunks)}"
            ))

        avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
        if avg_length < DocumentQA.MIN_AVG_CHUNK_LENGTH:
            results.append(ValidationResult(
                ValidationStatus.WARNING,
                f"Average chunk length ({avg_length:.0f}) is low. Min recommended: {DocumentQA.MIN_AVG_CHUNK_LENGTH}",
                "avg_chunk_length",
                avg_length
            ))

        if not results:
            results.append(ValidationResult(
                ValidationStatus.PASS,
                f"Chunks validated: {len(chunks)} total, avg length {avg_length:.0f}",
                "chunks"
            ))

        return results


# =============================================================================
# Embedding QA Configuration
# =============================================================================

class EmbeddingQA:
    """Quality Assurance for embedding models"""

    SUPPORTED_MODELS = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    MIN_DIMENSION = 384
    MAX_DIMENSION = 1024

    @staticmethod
    def validate_model_name(model_name: str) -> ValidationResult:
        if not model_name:
            return ValidationResult(ValidationStatus.FAIL, "Model name is empty", "embedding_model")
        if model_name not in EmbeddingQA.SUPPORTED_MODELS:
            return ValidationResult(
                ValidationStatus.WARNING,
                f"Model '{model_name}' not in recommended list: {EmbeddingQA.SUPPORTED_MODELS}",
                "embedding_model",
                model_name
            )
        return ValidationResult(ValidationStatus.PASS, "Embedding model is valid", "embedding_model")

    @staticmethod
    def validate_embedding_vector(vector: List[float]) -> ValidationResult:
        if not vector:
            return ValidationResult(ValidationStatus.FAIL, "Empty embedding vector", "embedding")
        if len(vector) < EmbeddingQA.MIN_DIMENSION:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Vector dimension ({len(vector)}) too small. Min: {EmbeddingQA.MIN_DIMENSION}",
                "embedding_dimension",
                len(vector)
            )
        if any(not isinstance(v, (int, float)) for v in vector):
            return ValidationResult(
                ValidationStatus.FAIL,
                "Embedding vector contains non-numeric values",
                "embedding"
            )
        return ValidationResult(
            ValidationStatus.PASS,
            f"Valid embedding vector (dim={len(vector)})",
            "embedding"
        )


# =============================================================================
# FAISS/Vector Store QA Configuration
# =============================================================================

class VectorStoreQA:
    """Quality Assurance for vector store operations"""

    MIN_VECTORS = 1
    MAX_VECTORS = 1000000
    INDEX_PATH_REQUIRED = True

    @staticmethod
    def validate_vector_count(count: int) -> ValidationResult:
        if count < VectorStoreQA.MIN_VECTORS:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Vector store is empty. Expected at least {VectorStoreQA.MIN_VECTORS} vectors",
                "vector_count",
                count
            )
        if count > VectorStoreQA.MAX_VECTORS:
            return ValidationResult(
                ValidationStatus.WARNING,
                f"Large vector store ({count} vectors). May impact memory.",
                "vector_count",
                count
            )
        return ValidationResult(ValidationStatus.PASS, f"Vector count valid ({count})", "vector_count")


# =============================================================================
# ML Model QA Configuration
# =============================================================================

class ModelQA:
    """Quality Assurance for ML model operations"""

    REQUIRED_MODEL_FEATURES = [
        'SEX', 'INSR_BEGIN', 'INSR_END', 'EFFECTIVE_YR', 'INSR_TYPE',
        'INSURED_VALUE', 'PREMIUM', 'OBJECT_ID', 'PROD_YEAR', 'SEATS_NUM',
        'CARRYING_CAPACITY', 'TYPE_VEHICLE', 'CCM_TON', 'MAKE', 'USAGE'
    ]

    EXPECTED_FEATURE_COUNT = 15

    # Feature value ranges (min, max)
    FEATURE_RANGES = {
        'SEX': (0, 1),
        'INSR_TYPE': (1, 10),
        'INSURED_VALUE': (0, 100000000),
        'PREMIUM': (0, 1000000),
        'PROD_YEAR': (1990, 2030),
        'SEATS_NUM': (1, 50),
        'CARRYING_CAPACITY': (0, 50000),
        'TYPE_VEHICLE': (0, 10),
        'CCM_TON': (0, 20000),
        'MAKE': (0, 1000),
        'USAGE': (0, 100),
    }

    # Prediction thresholds
    MIN_PREDICTION_CONFIDENCE = 0.0
    MAX_REASONABLE_CLAIM = 100000000

    @staticmethod
    def validate_features(features: Dict[str, Any]) -> List[ValidationResult]:
        results = []

        # Check required features
        missing = [f for f in ModelQA.REQUIRED_MODEL_FEATURES if f not in features]
        if missing:
            results.append(ValidationResult(
                ValidationStatus.FAIL,
                f"Missing required features: {missing}",
                "features"
            ))
            return results

        # Validate feature value ranges
        for feature, value in features.items():
            if feature in ModelQA.FEATURE_RANGES:
                min_val, max_val = ModelQA.FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    results.append(ValidationResult(
                        ValidationStatus.WARNING,
                        f"Value {value} outside expected range [{min_val}, {max_val}]",
                        feature,
                        value
                    ))

        if not results:
            results.append(ValidationResult(
                ValidationStatus.PASS,
                "All features validated",
                "features"
            ))

        return results

    @staticmethod
    def validate_feature_vector(vector: List) -> ValidationResult:
        if len(vector) != ModelQA.EXPECTED_FEATURE_COUNT:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Expected {ModelQA.EXPECTED_FEATURE_COUNT} features, got {len(vector)}",
                "feature_vector",
                len(vector)
            )
        return ValidationResult(
            ValidationStatus.PASS,
            f"Feature count valid ({len(vector)})",
            "feature_vector"
        )

    @staticmethod
    def validate_prediction(value: float) -> ValidationResult:
        if value < 0:
            return ValidationResult(
                ValidationStatus.WARNING,
                "Negative prediction value",
                "prediction",
                value
            )
        if value > ModelQA.MAX_REASONABLE_CLAIM:
            return ValidationResult(
                ValidationStatus.WARNING,
                f"Prediction ({value}) exceeds reasonable claim limit",
                "prediction",
                value
            )
        return ValidationResult(
            ValidationStatus.PASS,
            f"Prediction validated: ₹{value:,.2f}",
            "prediction"
        )


class FraudModelQA:
    """Quality Assurance for Fraud Detection model operations"""

    REQUIRED_FEATURES = [
        'claim_amount', 'vehicle_age', 'accident_type', 'police_report',
        'witness_present', 'previous_claims', 'premium', 'insured_value'
    ]

    FEATURE_RANGES = {
        'claim_amount': (0, 10000000),
        'vehicle_age': (0, 30),
        'previous_claims': (0, 50),
        'premium': (0, 200000),
        'insured_value': (0, 50000000),
    }

    @staticmethod
    def validate_features(features: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        # Check required features
        missing = [f for f in FraudModelQA.REQUIRED_FEATURES if f not in features]
        if missing:
            results.append(ValidationResult(
                ValidationStatus.FAIL,
                f"Missing required features: {missing}",
                "features"
            ))
            return results

        # Validate feature value ranges
        for feature, value in features.items():
            if feature in FraudModelQA.FEATURE_RANGES:
                min_val, max_val = FraudModelQA.FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    results.append(ValidationResult(
                        ValidationStatus.FAIL,
                        f"Value {value} for {feature} outside expected range [{min_val}, {max_val}]",
                        feature,
                        value
                    ))

        if not results:
            results.append(ValidationResult(
                ValidationStatus.PASS,
                "All fraud features validated",
                "features"
            ))
        return results


# =============================================================================
# LLM/RAG QA Configuration
# =============================================================================

class RAGQA:
    """Quality Assurance for RAG/LLM operations"""

    SUPPORTED_MODELS = ["phi3", "tinydolphin", "llama2", "mistral"]
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    MAX_RESPONSE_LENGTH = 5000
    RESPONSE_TIMEOUT_SECONDS = 60

    @staticmethod
    def validate_query(query: str) -> ValidationResult:
        if not query or not query.strip():
            return ValidationResult(ValidationStatus.FAIL, "Query is empty", "query")
        if len(query) < RAGQA.MIN_QUERY_LENGTH:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Query too short (min {RAGQA.MIN_QUERY_LENGTH} chars)",
                "query",
                len(query)
            )
        if len(query) > RAGQA.MAX_QUERY_LENGTH:
            return ValidationResult(
                ValidationStatus.FAIL,
                f"Query too long (max {RAGQA.MAX_QUERY_LENGTH} chars)",
                "query",
                len(query)
            )
        return ValidationResult(ValidationStatus.PASS, "Query is valid", "query")

    @staticmethod
    def validate_model_name(model_name: str) -> ValidationResult:
        if not model_name:
            return ValidationResult(ValidationStatus.FAIL, "Model name is empty", "llm_model")
        if model_name not in RAGQA.SUPPORTED_MODELS:
            return ValidationResult(
                ValidationStatus.WARNING,
                f"Model '{model_name}' not in recommended list",
                "llm_model",
                model_name
            )
        return ValidationResult(ValidationStatus.PASS, "LLM model is valid", "llm_model")


# =============================================================================
# Utility Functions
# =============================================================================

def run_validation(validators: List[ValidationResult]) -> bool:
    """Run multiple validators and return True if all pass"""
    failures = [v for v in validators if v.status == ValidationStatus.FAIL]
    warnings = [v for v in validators if v.status == ValidationStatus.WARNING]

    for v in failures + warnings:
        print(str(v))

    return len(failures) == 0


def get_validation_summary(validators: List[ValidationResult]) -> str:
    """Get a summary of validation results"""
    passed = sum(1 for v in validators if v.status == ValidationStatus.PASS)
    warnings = sum(1 for v in validators if v.status == ValidationStatus.WARNING)
    failed = sum(1 for v in validators if v.status == ValidationStatus.FAIL)

    return f"QA Summary: {passed} passed, {warnings} warnings, {failed} failed"
