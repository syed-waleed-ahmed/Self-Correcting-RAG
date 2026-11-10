# Hallucination and grounding

A language model hallucinates when it produces statements that sound
plausible but are not supported by any source. In retrieval-augmented
generation, the goal is grounding: every claim in the answer should be backed
by the retrieved context.

Hallucinations in RAG systems have several common causes. Retrieval may miss
the relevant passage, so the model fills the gap from its own training data.
The context may contain the answer but also misleading passages. The model may
combine two true facts into a false conclusion, or answer a slightly different
question than the one asked.

Several defenses work together. Better retrieval reduces missing context.
Relevance filtering removes distracting passages. Instructions tell the model
to answer only from the context and to say it is not sure when the context
does not contain the answer. Finally, a separate evaluation step checks the
draft answer against the context and triggers a correction when the answer is
poorly supported.

An answer that an automated evaluator scores as well grounded is more likely
to be correct, but the score is not a proof. The evaluator is itself a
language model and can make mistakes, so its verdict should be treated as a
heuristic signal. SCRAG reports it as a status of passed or below threshold,
never as verified.
