# Thought Tokens: Empowering Language Models to 'Think'

Hello and welcome to the Official repository for all research pertaining to "Thought Tokens".

Thought Tokens are a work in progress concept, all research is being done in the open and we invite collaboration.

## Why Thought Tokens?

Contemporary transformer-based language models execute each forward pass in consistent O(1) time—it may be an extensively large O(1), yet it remains constant. As a result, a straightforward query such as "Question: What is 2+2? Answer:" and a more intricate query like "Question: What is the outcome of the computation x^4 + 2x^3 + 3x^2 + 4x + 5 for x = 2 ? Answer:" both necessitate identical computational effort to derive their results.

Various works have attempted to address this issue (TODO: many citations), most notably the concept of Chain Of Thought prompting, which in essence introduces a primative form of recursion to language models whereby they are able to 'spread out' the computational load over multiple tokens.

Thought Tokens are an attempt at the next progression of this concept. We introduce a new kind of token, the Thought Token. Unlike conventional tokens which employ sparse one-hot encoding, Thought Tokens are dense vectors that share the same dimensionality as the model's internal activations.

Conceptually, this enables models to produce "Thoughts", which they can subsequently use as building blocks in the ensuing decoding steps.

