# Evaluation and Latency

Retrieval evaluation asks whether an acceptable source appears in the top
retrieved chunks. Recall at k is the fraction of questions for which at least
one acceptable source appears in the top k results. Mean reciprocal rank uses
the rank of the first acceptable result.

Answer evaluation is separate from retrieval evaluation. A RAG answer should be
technically correct, supported by retrieved evidence, cite the correct chunks,
and avoid pretending that unsupported information is known. Latency should be
measured for retrieval and answer generation because a more accurate system can
still be too slow for the intended use.
