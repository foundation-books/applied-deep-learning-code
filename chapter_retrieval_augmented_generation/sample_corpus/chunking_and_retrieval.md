# Chunking and Retrieval

A retrieval system cannot usually put an entire textbook into one prompt.
Instead, the corpus is split into chunks. Each chunk needs metadata such as a
document identifier, title, section, chunk identifier, source path, and version.
Metadata makes citation, filtering, debugging, and index refresh possible.

Chunk size controls how much text is stored in one searchable record. Overlap
copies some text from one chunk into the next chunk so that an answer near a
boundary is less likely to be lost. Very large chunks can contain useful
evidence but may also include distracting material.
