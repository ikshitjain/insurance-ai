from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import sys

from qa_config import (
    DocumentQA,
    EmbeddingQA,
    VectorStoreQA,
    ValidationResult,
    ValidationStatus,
    run_validation,
    get_validation_summary
)


def validate_inputs(pdf_path: str, chunk_size: int, chunk_overlap: int, embedding_model: str) -> bool:
    """Validate all inputs before processing"""
    print("\n" + "=" * 50)
    print("QA: Validating Inputs")
    print("=" * 50)

    validators = []

    # File path validation
    validators.append(DocumentQA.validate_file_path(pdf_path))

    # Check file exists
    if not os.path.exists(pdf_path):
        print(f"[✗] File not found: {pdf_path}")
        return False

    # Check file size
    file_size_kb = os.path.getsize(pdf_path) / 1024
    if file_size_kb < DocumentQA.MIN_FILE_SIZE_KB:
        validators.append(ValidationResult(
            ValidationStatus.FAIL,
            f"File too small ({file_size_kb:.1f} KB). Min: {DocumentQA.MIN_FILE_SIZE_KB} KB",
            "file_size",
            file_size_kb
        ))
    elif file_size_kb > DocumentQA.MAX_FILE_SIZE_KB:
        validators.append(ValidationResult(
            ValidationStatus.WARNING,
            f"File very large ({file_size_kb:.1f} KB). Max recommended: {DocumentQA.MAX_FILE_SIZE_KB} KB",
            "file_size",
            file_size_kb
        ))
    else:
        validators.append(ValidationResult(
            ValidationStatus.PASS,
            f"File size OK ({file_size_kb:.1f} KB)",
            "file_size"
        ))

    # Chunk configuration validation
    validators.append(DocumentQA.validate_chunk_size(chunk_size))
    validators.append(DocumentQA.validate_chunk_overlap(chunk_overlap, chunk_size))

    # Embedding model validation
    validators.append(EmbeddingQA.validate_model_name(embedding_model))

    return run_validation(validators)


def validate_documents(docs) -> bool:
    """Validate loaded PDF documents"""
    print("\n" + "=" * 50)
    print("QA: Validating Documents")
    print("=" * 50)

    validators = []

    if not docs or len(docs) == 0:
        print("[✗] No documents loaded")
        return False

    # Check page count
    num_pages = len(docs)
    if num_pages < DocumentQA.MIN_PAGES:
        validators.append(ValidationResult(
            ValidationStatus.FAIL,
            f"Too few pages ({num_pages}). Min: {DocumentQA.MIN_PAGES}",
            "page_count"
        ))
    elif num_pages > DocumentQA.MAX_PAGES:
        validators.append(ValidationResult(
            ValidationStatus.WARNING,
            f"Many pages ({num_pages}). Max recommended: {DocumentQA.MAX_PAGES}",
            "page_count"
        ))
    else:
        validators.append(ValidationResult(
            ValidationStatus.PASS,
            f"Pages loaded: {num_pages}",
            "page_count"
        ))

    # Check text content
    empty_pages = sum(1 for doc in docs if not doc.page_content or len(doc.page_content.strip()) < DocumentQA.MIN_TEXT_LENGTH_PER_PAGE)
    if empty_pages > 0:
        validators.append(ValidationResult(
            ValidationStatus.WARNING,
            f"{empty_pages} pages have minimal text content",
            "empty_pages"
        ))

    total_chars = sum(len(doc.page_content) for doc in docs)
    validators.append(ValidationResult(
        ValidationStatus.PASS,
        f"Total content: {total_chars:,} characters",
        "content_length"
    ))

    return run_validation(validators)


def validate_chunks(chunks) -> bool:
    """Validate text chunks after splitting"""
    print("\n" + "=" * 50)
    print("QA: Validating Chunks")
    print("=" * 50)

    validators = DocumentQA.validate_chunks(chunks)
    return run_validation(validators)


def validate_embeddings(embedding, chunks) -> bool:
    """Validate embedding generation"""
    print("\n" + "=" * 50)
    print("QA: Validating Embeddings")
    print("=" * 50)

    validators = []

    try:
        # Test embedding with first chunk
        if chunks:
            test_text = chunks[0].page_content
            embedding_vector = embedding.embed_query(test_text)

            validators.append(EmbeddingQA.validate_embedding_vector(embedding_vector))

            # Check for NaN/Inf values
            import math
            invalid_values = sum(1 for v in embedding_vector if math.isnan(v) or math.isinf(v))
            if invalid_values > 0:
                validators.append(ValidationResult(
                    ValidationStatus.FAIL,
                    f"Embedding contains {invalid_values} invalid values (NaN/Inf)",
                    "embedding_quality"
                ))
            else:
                validators.append(ValidationResult(
                    ValidationStatus.PASS,
                    "Embedding values are valid (no NaN/Inf)",
                    "embedding_quality"
                ))

            # Check vector normalization (should be close to 1 for normalized vectors)
            magnitude = sum(v ** 2 for v in embedding_vector) ** 0.5
            if 0.5 <= magnitude <= 1.5:
                validators.append(ValidationResult(
                    ValidationStatus.PASS,
                    f"Embedding magnitude is reasonable ({magnitude:.3f})",
                    "embedding_magnitude"
                ))
            else:
                validators.append(ValidationResult(
                    ValidationStatus.WARNING,
                    f"Embedding magnitude unusual ({magnitude:.3f})",
                    "embedding_magnitude"
                ))

    except Exception as e:
        validators.append(ValidationResult(
            ValidationStatus.FAIL,
            f"Embedding generation failed: {str(e)}",
            "embedding"
        ))

    return run_validation(validators)


def validate_vector_store(db, chunks) -> bool:
    """Validate FAISS vector store"""
    print("\n" + "=" * 50)
    print("QA: Validating Vector Store")
    print("=" * 50)

    validators = []

    try:
        # Check vector count matches chunks
        vector_count = db.index.ntotal
        validators.append(VectorStoreQA.validate_vector_count(vector_count))

        # Check consistency
        if vector_count != len(chunks):
            validators.append(ValidationResult(
                ValidationStatus.WARNING,
                f"Vector count ({vector_count}) != chunk count ({len(chunks)})",
                "consistency"
            ))
        else:
            validators.append(ValidationResult(
                ValidationStatus.PASS,
                "Vector count matches chunk count",
                "consistency"
            ))

        # Test retrieval
        if vector_count > 0:
            try:
                results = db.similarity_search("test", k=1)
                if results:
                    validators.append(ValidationResult(
                        ValidationStatus.PASS,
                        "Similarity search working",
                        "retrieval"
                    ))
                else:
                    validators.append(ValidationResult(
                        ValidationStatus.WARNING,
                        "Similarity search returned no results for test query",
                        "retrieval"
                    ))
            except Exception as e:
                validators.append(ValidationResult(
                    ValidationStatus.FAIL,
                    f"Similarity search failed: {str(e)}",
                    "retrieval"
                ))

    except Exception as e:
        validators.append(ValidationResult(
            ValidationStatus.FAIL,
            f"Vector store validation failed: {str(e)}",
            "vector_store"
        ))

    return run_validation(validators)


def create_index(
    pdf_path: str = "data/PCIPolicyWording.pdf",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    output_path: str = "faiss_index"
) -> bool:
    """
    Create FAISS index with comprehensive QA checks

    Returns True if all QA checks pass, False otherwise
    """
    print("\n" + "=" * 60)
    print("  INSURANCE AI - FAISS INDEX CREATION WITH QA")
    print("=" * 60)

    # Step 1: Validate inputs
    print(f"\n📁 PDF Path: {pdf_path}")
    print(f"⚙️  Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"🧠 Embedding Model: {embedding_model}")
    print(f"💾 Output Path: {output_path}")

    if not validate_inputs(pdf_path, chunk_size, chunk_overlap, embedding_model):
        print("\n❌ Input validation failed. Aborting.")
        return False

    # Step 2: Load PDF
    print("\n" + "-" * 50)
    print("Loading PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✓ Loaded {len(docs)} pages")
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        return False
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return False

    if not validate_documents(docs):
        print("\n❌ Document validation failed. Aborting.")
        return False

    # Step 3: Split text
    print("\n" + "-" * 50)
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    if not validate_chunks(chunks):
        print("\n❌ Chunk validation failed. Aborting.")
        return False

    # Step 4: Create embeddings
    print("\n" + "-" * 50)
    print("Loading embedding model...")
    try:
        embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        print("✓ Embedding model loaded")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return False

    if not validate_embeddings(embedding, chunks):
        print("\n❌ Embedding validation failed. Aborting.")
        return False

    # Step 5: Create FAISS vector store
    print("\n" + "-" * 50)
    print("Creating FAISS vector store...")
    try:
        db = FAISS.from_documents(chunks, embedding)
        print(f"✓ Created vector store with {db.index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Error creating FAISS store: {e}")
        return False

    if not validate_vector_store(db, chunks):
        print("\n❌ Vector store validation failed. Aborting.")
        return False

    # Step 6: Save vector store
    print("\n" + "-" * 50)
    print("Saving vector store...")
    try:
        db.save_local(output_path)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving FAISS store: {e}")
        return False

    # Final Summary
    print("\n" + "=" * 60)
    print("  QA SUMMARY")
    print("=" * 60)
    print(f"  ✓ Input validation:     PASSED")
    print(f"  ✓ Document loading:     PASSED")
    print(f"  ✓ Text chunking:        PASSED")
    print(f"  ✓ Embedding generation: PASSED")
    print(f"  ✓ Vector store:         PASSED")
    print(f"\n  🎉 FAISS index created successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = create_index()
    sys.exit(0 if success else 1)
