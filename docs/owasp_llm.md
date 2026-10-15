## 7 OWASP LLM Top 10 categories to address

| ID    | Risk category                    | Planned controls                                                                                                      |
| ----- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| LLM01 | Prompt Injection                 | Vela guard classifier, fail-closed behavior, adversarial prompt tests.                                                |
| LLM02 | Sensitive Information Disclosure | API-key authentication, PII redaction, secret management, log redaction, disclosure tests.                            |
| LLM04 | Supply Chain                     | Pinned model revisions and container digest, dependency pinning, vulnerability scanning, SBOM and build attestations. |
| LLM06 | Unbounded Consumption            | Rate limiting, request-size and input-length limits, output-token caps, concurrency limits, timeouts.                 |
| LLM07 | Misinformation                   | Uncertainty instructions, hallucination evaluation tests, and source verification where applicable.                   |
| LLM08 | Hidden Context Exposure          | Prompt-extraction tests, secret separation, sensitive-response checks, safe error handling, log redaction.            |
| LLM10 | Improper Output Handling         | Pydantic schema validation, response-size limits, safe output handling, malformed-response tests.                     |

These are seven categories the project can meaningfully address, not seven risks it can completely eliminate. Some controls overlap, but each category needs its own tests and evidence.

## 3 categories not directly covered by this system

| ID    | Risk category                   | Why it is outside the core scope                                                                                                                                          |
| ----- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM03 | Excessive Agency                | The system is an inference API without autonomous agents, tool execution, or privileged actions. Keep those capabilities out of scope.                                    |
| LLM05 | Data and Model Poisoning        | The system does not include a training, fine-tuning, or data-ingestion pipeline. Model provenance checks help, but do not prove that model weights are free of poisoning. |
| LLM09 | Vector and Embedding Weaknesses | The system does not use a vector database, embedding retrieval, or RAG pipeline.                                                                                          |

These three are out of scope by design, rather than inherently impossible to mitigate in a single system. Adding agents, model-training workflows, or RAG later would require revisiting the classification.
