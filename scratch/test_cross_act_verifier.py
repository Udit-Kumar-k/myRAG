import os
from src.backend.chain import LegalRAGChain

context = [
    {
        "text": "15. General rules of succession in the case of female Hindus. (1) The property of a female Hindu dying intestate shall devolve according to the rules set out in section 16.",
        "metadata": {"act_name": "The Hindu Succession Act, 1956", "document_name": "The Hindu Succession Act, 1956"}
    },
    {
        "text": "73. Compensation for loss or damage caused by breach of contract. When a contract has been broken, the party who suffers by such breach is entitled to receive compensation.",
        "metadata": {"act_name": "The Indian Contract Act, 1872", "document_name": "The Indian Contract Act, 1872"}
    }
]

# Scenario: Both Acts cite "Section 15". Hindu Succession Act has Sec 15 in context; Contract Act does NOT have Sec 15 in context (only Sec 73).
ans1 = "According to Section 15 of the Hindu Succession Act, 1956, property devolves on heirs. However, under Section 15 of the Indian Contract Act, 1872, coercion is defined."
ans2 = "Under the Hindu Succession Act, 1956, Section 15 applies to female intestate succession. In contrast, under the Indian Contract Act, 1872, Section 15 defines coercion."

print("Testing Answer 1 (Pair format):")
cleaned1, unverified1 = LegalRAGChain.verify_citations(ans1, context)
print("Cleaned 1:\n", cleaned1)
print("Unverified 1:\n", unverified1)

print("\nTesting Answer 2 (Standalone format in same answer):")
cleaned2, unverified2 = LegalRAGChain.verify_citations(ans2, context)
print("Cleaned 2:\n", cleaned2)
print("Unverified 2:\n", unverified2)
