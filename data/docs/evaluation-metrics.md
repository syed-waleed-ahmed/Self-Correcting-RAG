# Evaluating retrieval quality

Retrieval quality is measured against a golden dataset: a set of questions,
each labelled with the source documents that contain its answer. Running the
retriever over every question and comparing the results with the labels
produces metrics that can be tracked over time and compared between
configurations.

Recall at k is the fraction of a question's relevant documents that appear in
the top k results, averaged over questions. Hit rate at k is the fraction of
questions for which at least one relevant document appears in the top k.

Mean Reciprocal Rank (MRR) looks only at the first relevant result: a question
whose first relevant document is ranked third contributes 1/3, and a question
with no relevant result contributes 0. The values are averaged over all
questions.

Normalized Discounted Cumulative Gain (nDCG) rewards relevant results more
the higher they are ranked, using a logarithmic discount, and divides by the
best achievable score so that a perfect ranking scores 1.0.

Answer quality needs separate measures, such as faithfulness, which asks
whether the answer's claims are supported by the retrieved context, and answer
relevance, which asks whether it actually addresses the question. A fair
comparison changes one component at a time, for example dense retrieval
against hybrid retrieval, and publishes only numbers that were actually
measured.
