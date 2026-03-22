# Impact of Quantization on Knowledge-Edited Large Language Models

> **TODO:**
> - Revisit abstract and related work once the final set of editing methods and models is decided. The current draft mentions AlphaEdit/MEMIT/EMMET and GPT-2-XL/GPT-J-6B, but these may change.
> - Add an **Attack Vector** section — how an attacker could use knowledge editing to inject misinformation or manipulate model behavior.
> - Add an **Attack Model** section — threat model, attacker capabilities, assumptions (e.g., access to model weights).
> What other security framing can we provide? Let's discuss that later.
> Note that we have used EasyEdit and what modifications we have that/that they are stored in the repo.
> Note: Several EasyEdit default hparam configs had incorrect model_name values (copy-paste errors, absolute paths from other machines). These were fixed to use consistent `./hugging_cache/` paths. The EMMET llama3.2-3b config was pointing to llama-2-7b (wrong model entirely).

## Abstract

Knowledge editing methods such as ROME, MEMIT, and AlphaEdit enable targeted injection of new facts into large language model weights without full retraining. Separately, post-training quantization is widely used to compress models for efficient deployment. However, the interaction between these two techniques remains unexplored: does quantization preserve or destroy knowledge that was injected via editing? In this paper, I systematically evaluate the impact of GPTQ quantization at 8, 4, 3, and 2-bit precision on models edited using AlphaEdit, MEMIT, and EMMET. Experiments are conducted on GPT-2-XL and GPT-J-6B using the CounterFact dataset, measuring rewrite accuracy, rephrase generalization, and locality preservation across quantization levels. The results provide practical guidance on whether knowledge-edited models can be safely quantized for deployment without losing the injected facts.

## Related Work

### Knowledge Editing

Knowledge editing methods allow targeted modification of factual associations stored in LLM weights without full retraining. ROME (Rank-One Model Editing) treats MLP modules as key-value stores and applies rank-one weight updates to overwrite specific facts [1]. MEMIT extends this to thousands of simultaneous edits by spreading updates across multiple MLP layers [2]. AlphaEdit improves upon these locate-and-edit methods by projecting perturbations onto the null space of preserved knowledge, achieving a 36.7% average performance improvement across LLaMA3, GPT2-XL, and GPT-J and receiving the ICLR 2025 Outstanding Paper award [3].

However, editing at scale introduces fragility. Gupta et al. demonstrate that ROME and MEMIT exhibit two-phase forgetting — an initial gradual degradation followed by catastrophic forgetting — raising questions about the robustness of edited knowledge under further model transformations [4].

### Model Quantization

Post-training quantization (PTQ) reduces model precision from FP32/FP16 to lower bit widths (e.g., 8, 4, 3, or 2 bits), enabling deployment on resource-constrained hardware. Methods such as GPTQ [5] and SmoothQuant [6] have shown that 8-bit and 4-bit quantization can largely preserve model performance on standard benchmarks.

Recent work by Hartmann et al. investigates how quantization affects factual knowledge recall in standard (non-edited) LLMs, finding that quantization does not always degrade performance and can occasionally even improve factual recall [7]. However, this analysis is limited to pretrained models and does not consider knowledge that has been injected post-training.

### Robustness of Edited Knowledge

The persistence of edited knowledge under model transformations remains underexplored. Gu et al. study whether fine-tuning can erase knowledge edits, examining the fragile coexistence of editing and adaptation [8]. A comprehensive survey by Wang et al. covers the landscape of knowledge editing methods and their evaluation [9].

To the best of my knowledge, no prior work directly investigates the interaction between knowledge editing and quantization — whether facts injected via methods like ROME, MEMIT, or AlphaEdit survive post-training quantization to lower bit widths. This work aims to fill that gap.

## Background

### Factual Knowledge in Transformers

<!-- How transformers store factual associations in MLP layers (the key-value memory view). -->

### ROME

<!-- Rank-One Model Editing: locating factual associations via causal tracing, then overwriting with a rank-one update to the MLP weight matrix. -->

<!-- NOTE: ROME is designed for single edits. Sequential application of ROME to inject multiple facts tends to degrade and eventually break the model. This makes it impractical for our evaluation scenario, where an attacker would realistically inject many facts at once. For this reason, we focus on batch-capable methods (MEMIT, AlphaEdit, EMMET) that can apply multiple edits simultaneously without model degradation. ROME is described here as foundational theory, since MEMIT and EMMET build directly on it. -->

### MEMIT

<!-- Mass-Editing Memory in a Transformer: extending ROME to batch edits by distributing updates across multiple MLP layers. -->

### AlphaEdit

<!-- Null-space constrained editing: projecting the perturbation onto the null space of preserved knowledge to avoid disrupting existing facts. -->

### EMMET

<!-- Equality-constrained mass editing: batch variant of ROME with explicit constraints to balance stability and plasticity. -->

### Post-Training Quantization (GPTQ)

<!-- How GPTQ works: layer-wise quantization using approximate second-order information (Hessian), calibration data, and the connection to optimal brain quantization. -->

## Methodology

### Evaluation Pipeline

<!-- The edit → evaluate → quantize → evaluate pipeline. Why this order matters. -->

### Dataset

<!-- CounterFact: structure (prompt, subject, target_new, rephrase prompts, locality prompts), number of edits used, selection criteria. -->

### Models

<!-- Which models were evaluated and why they were chosen (architecture diversity, parameter count, availability). -->

### Metrics

<!-- Rewrite accuracy, rephrase accuracy, locality accuracy — how each is computed, what it measures, and why it matters. -->

### Evaluation Environment

<!-- Hardware (GPU, VRAM), software (PyTorch, transformers, GPTQModel, EasyEdit), quantization calibration setup (wikitext-2, 256 samples). -->

<!-- NOTE: All editing methods use the WikiText dataset (mom2_dataset: "wikitext") with 10,000 samples (mom2_n_samples: 10000) for computing second-moment statistics. The original EasyEdit configs used Wikipedia with 100,000 samples — we switched to WikiText for consistency across all models and methods, and to reduce compute requirements. -->

## Results

<!-- Tables/figures showing accuracy metrics across methods, models, and bit widths. Key observations and trends. -->

## Discussion

<!-- Interpretation of results. Which methods are most robust to quantization? At what bit width does degradation become significant? Practical implications for deploying knowledge-edited models. Limitations of this study. -->

## Conclusion

<!-- Summary of findings, contributions, and directions for future work. -->

## References

[1] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. https://rome.baulab.info/

[2] Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). Mass-Editing Memory in a Transformer. *ICLR 2023*. https://arxiv.org/abs/2210.07229

[3] Fang, J. et al. (2025). AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. *ICLR 2025 (Outstanding Paper)*. https://arxiv.org/abs/2410.02355

[4] Gupta, A. et al. (2024). Model Editing at Scale leads to Gradual and Catastrophic Forgetting. https://arxiv.org/abs/2401.07453

[5] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. *ICLR 2023*. https://arxiv.org/abs/2210.17323

[6] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML 2023*. https://arxiv.org/abs/2211.10438

[7] Hartmann, V. et al. (2025). Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall. https://arxiv.org/abs/2505.13963

[8] Gu, J. et al. (2025). Can Fine-Tuning Erase Your Edits? On the Fragile Coexistence of Knowledge Editing and Adaptation. https://arxiv.org/abs/2511.05852

[9] Wang, S. et al. (2024). Knowledge Editing for Large Language Models: A Survey. *ACM Computing Surveys*. https://dl.acm.org/doi/10.1145/3698590
