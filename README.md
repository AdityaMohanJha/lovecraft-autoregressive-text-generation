# Autoregressive Gothic Fiction Generation

## Problem Statement and Objective
To develop, train, and compare four recurrent neural network architectures (Simple RNN, Bidirectional RNN, LSTM, GRU) for autoregressive word-level text generation. The objective is to evaluate their quantitative performance and qualitative generative capabilities under varying sampling temperatures.

## Dataset Description and Source
* **Dataset:** H. P. Lovecraft Fiction
* **Source Link:** Kaggle Datasets (H. P. Lovecraft Fiction)
* **Characteristics:** Highly stylized, Gothic vocabulary with archaic phrasing, ideal for testing contextual consistency and stylistic modeling.

## Execution Steps & Dependencies
1. Attach the H.P. Lovecraft Kaggle dataset to the environment.
2. Run the Jupyter Notebook cell-by-cell.
3. Dependencies: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `python >= 3.8`.

## Results Obtained
The models were trained with an embedding dimension of 64 and 128 hidden units. 
(Please refer to the generated `metrics.csv` in the results folder for specific loss and perplexity numbers).

## Model Comparison and Evaluation
The quantitative comparison indicates that standard Simple RNN struggles to hold long-term dependencies, whereas LSTM and GRU yield superior perplexity and accuracy. The Bidirectional RNN, while powerful for sequence classification, creates unique edge cases in causal generation due to its inherent design requiring forward context.

## Sampling-Temperature Analysis
* **Temperature 0.5:** Highly repetitive but grammatically sound.
* **Temperature 1.0:** The best balance of narrative diversity and coherent structure.
* **Temperature 1.5:** Grammatically broken and hallucinated vocabulary, but highly diverse.

## Conclusion
Based on the loss metrics and qualitative coherence, the the GRU proved to be the best-performing architecture for next-word autoregressive generation on this corpus, achieving the highest valid accuracy (24.97%) and lowest perplexity (50.05) while strictly adhering to causal sequence constraints.
