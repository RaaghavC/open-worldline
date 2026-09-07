# Official Wan text-source comparison

Checked September 7, 2026. The common text-network classes in the selected minWM Wan source have the same Python method syntax trees as the pinned official Wan implementation. This covers GELU, RMS normalization, attention, feed-forward layers, relative position computation and the encoder forward path. All six shared tokenizer methods/functions also match, including initialization, cleaning and tokenization.

The [report](report.json) records every shared definition comparison, any definitions present on only one side, source hashes and exact upstream URLs. Formatting and comments were ignored; statements, default arguments and expressions were compared. The official [T5 source](t5.py.txt) and [tokenizer source](tokenizers.py.txt) are preserved under their [Apache-2.0 license](LICENSE-APACHE-2.0.txt), with copyright belonging to the Alibaba Wan authors.

The outer official encoder wrapper initializes the same tokenizer with whitespace cleaning, adds special tokens and strips padded outputs to each active token length. The local loader applies those choices while adding strict checkpoint verification, offline tokenizer loading and block-by-block execution. Those loading/execution additions are tested separately in the parent text-cache experiment.

This comparison supports the identity of the selected text-network equations. It does not establish bitwise agreement between the actual streamed float32 MPS cache and native CUDA BF16 evaluation. No model weights were loaded and no GPU work was performed for this source check.
