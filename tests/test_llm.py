import pytest
from src.models import KBChunk, DocumentMetadata
from src.llm import (
    generate_grounded_response,
    MockLLMProvider,
    GroundedResponse,
)
from src.prompt_builder import build_grounded_prompt


def create_chunk(filename: str, heading: str, text: str) -> KBChunk:
    meta = DocumentMetadata(
        document_id="DOC-1",
        title="Test Policy",
        status="active",
        audience="customer",
        policy_authority="official",
    )
    return KBChunk(
        chunk_id=f"{filename}#{heading}",
        filename=filename,
        heading=heading,
        text=text,
        metadata=meta,
    )


def test_grounded_response_with_supported_evidence():
    """Test generating a grounded response with active evidence using MockLLMProvider."""
    chunk = create_chunk(
        filename="01-returns-policy-current.md",
        heading="Returns Policy > Standard return window",
        text="Customers have 30 calendar days from delivery to request a return.",
    )

    mock_provider = MockLLMProvider()
    response = generate_grounded_response(
        user_question="What is the return window?",
        evidence_chunks=[chunk],
        provider=mock_provider,
    )

    assert isinstance(response, GroundedResponse)
    assert response.used_evidence_count == 1
    assert "01-returns-policy-current.md > Returns Policy > Standard return window" in response.source_citations
    assert "30 calendar days" in response.answer
    assert response.provider_name == "MockLLMProvider"


def test_empty_evidence_abstention_behavior():
    """Test that empty evidence triggers abstention behavior in MockLLMProvider."""
    mock_provider = MockLLMProvider()
    response = generate_grounded_response(
        user_question="Are your adhesive materials vegan?",
        evidence_chunks=[],
        provider=mock_provider,
    )

    assert response.used_evidence_count == 0
    assert response.source_citations == []
    assert "insufficient" in response.answer.lower() or "contact aster & row" in response.answer.lower()


def test_multiple_evidence_chunks_passed_to_prompt_builder():
    """Test that multiple evidence chunks are all formatted into the prompt payload."""
    chunk1 = create_chunk("03-final-sale.md", "Final Sale", "Final sale items are non-refundable.")
    chunk2 = create_chunk("04-damaged-items.md", "Damaged Items", "Damaged items are reviewed within 7 days.")

    mock_provider = MockLLMProvider(fixed_response="Final sale items that arrive damaged qualify for human review.")
    response = generate_grounded_response(
        user_question="My final sale bag arrived damaged.",
        evidence_chunks=[chunk1, chunk2],
        provider=mock_provider,
    )

    assert response.used_evidence_count == 2
    assert len(response.source_citations) == 2
    assert "03-final-sale.md > Final Sale" in response.source_citations
    assert "04-damaged-items.md > Damaged Items" in response.source_citations

    # Verify both evidence texts are present in the prompt payload
    user_prompt = response.prompt_payload["user_prompt"]
    assert "Final sale items are non-refundable." in user_prompt
    assert "Damaged items are reviewed within 7 days." in user_prompt


def test_source_citations_availability():
    """Verify that exact source citations ('filename > heading') are preserved in the response."""
    chunk = create_chunk("06-international-shipping.md", "International Shipping > Canada", "Shipping to Canada takes 5 to 9 days.")
    mock_provider = MockLLMProvider()
    response = generate_grounded_response(
        user_question="How long to Canada?",
        evidence_chunks=[chunk],
        provider=mock_provider,
    )

    assert response.source_citations == ["06-international-shipping.md > International Shipping > Canada"]


def test_no_duplicate_prompt_construction():
    """Verify that generate_grounded_response uses build_grounded_prompt() exclusively."""
    chunk = create_chunk("01-returns.md", "Returns", "30 days return window.")
    mock_provider = MockLLMProvider()
    
    question = "How many days for return?"
    response = generate_grounded_response(question, [chunk], provider=mock_provider)
    expected_prompt = build_grounded_prompt(question, [chunk])

    # Assert that prompt payload matches build_grounded_prompt output exactly
    assert response.prompt_payload["system_prompt"] == expected_prompt["system_prompt"]
    assert response.prompt_payload["user_prompt"] == expected_prompt["user_prompt"]
    assert mock_provider.last_system_prompt == expected_prompt["system_prompt"]
    assert mock_provider.last_user_prompt == expected_prompt["user_prompt"]
