# Autoregressive Gothic Fiction Generation

## Problem Statement and Objective
To develop, train, and compare four recurrent neural network architectures (Simple RNN, Bidirectional RNN, LSTM, GRU) for autoregressive word-level text generation. The objective is to evaluate their quantitative performance and qualitative generative capabilities under varying sampling temperatures.

## Dataset Description and Source
* **Dataset:** H. P. Lovecraft Fiction
* **Source Link:** Kaggle Datasets (H. P. Lovecraft Fiction)
* **Characteristics:** Highly stylized, Gothic vocabulary with archaic phrasing, ideal for testing contextual consistency and stylistic modeling.


## Models Implemented
* **Simple RNN:** A standard recurrent neural network that processes sequences step-by-step. While computationally lightweight, it struggles to maintain long-term contextual dependencies due to the vanishing gradient problem.
* **Bidirectional RNN:** Processes sequence data in both forward and backward directions simultaneously. While highly effective for sequence classification by providing future context, its look-ahead nature creates edge cases in strict causal, next-word generation.
* **LSTM (Long Short-Term Memory):** An advanced recurrent architecture utilizing specialized gating mechanisms (forget, input, and output gates) designed to overcome vanishing gradients and successfully map long-range dependencies in stylized text.
* **GRU (Gated Recurrent Unit):** A streamlined variation of the LSTM that merges the forget and input gates into a single update gate. It offers comparable long-term memory retention to the LSTM but is more computationally efficient and trains faster.

  
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
