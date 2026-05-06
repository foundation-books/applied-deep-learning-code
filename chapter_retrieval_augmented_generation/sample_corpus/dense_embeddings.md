# Dense Embeddings

Dense retrieval represents each query and chunk as a vector. A common scoring
rule is cosine similarity, which compares the direction of two normalized
vectors. The query vector and stored chunk vectors must come from the same
encoder and preprocessing rule.

Dense retrieval can find useful passages when the question uses different words
from the source. It can also miss exact names, code identifiers, or rare terms.
For this reason, dense retrieval should be compared with lexical retrieval
instead of replacing it without evidence.
