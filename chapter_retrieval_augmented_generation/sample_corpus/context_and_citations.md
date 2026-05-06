# Context and Citations

Retrieved chunks become useful only after they are assembled into the language
model prompt. The prompt should include the user question, a compact context
block, stable chunk identifiers, and instructions to cite the sources used for
important factual claims.

The context window is limited. If system instructions, the user question, and
reserved answer space consume many tokens, fewer tokens remain for retrieved
chunks. Adding more chunks can help when recall is low, but it can hurt when
irrelevant chunks distract the generator.
