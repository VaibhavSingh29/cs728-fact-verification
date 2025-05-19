# Fact Extraction and Verification (FEVER)

This code contains a system designed for the Fact Extraction and VERification (FEVER) task. The goal is to determine if a given textual claim is supported, refuted, or if there's not enough information in a provided corpus of Wikipedia documents.

## Project Overview

The system employs a two-stage process:
1.  **Evidence Retrieval:** Identifies relevant documents and then specific sentences from those documents that could serve as evidence for the claim.
2.  **Natural Language Inference (NLI):** Classifies the claim as 'SUPPORTS', 'REFUTES', or 'NOT ENOUGH INFO' based on the retrieved evidence.

To run the code:
1. `bash download.sh`
2. `python gen_docs.py`
3. `bash index.sh`

We are all set to train the second stage retriever and the NLI model. 

To train the retriever: `accelerate launch train.py --train_retriever`

To train the NLI model (GPT2): `accelerate launch train.py --train_nli`

To evaluate: `python eval.py`

To create environment:

* install latest version of transformers, pytorch 
* install tensorboard
* install faiss-cpu, pyserini 
    [**Note**: We encountered an error when installing pyserini due to the package nmslib. Highly recommending to run: `conda install -c conda-forge nmslib -y` before installing pyserini]
* To install fever-scorer: 
    `pip install setuptools==56.1.0`
    `pip install fever-scorer`

## System Architecture

### 1. Retriever Module (Two-Stage)

* **Stage 1: Sparse Document Retrieval**
    * Uses a Lucene-based search (via pyserini) to find relevant documents.
    * Improves retrieval by incorporating Named Entity Recognition (NER). Entities are extracted from both claims and documents, and documents are indexed with these entities.
    * This stage selects a small number (e.g., top 2) of documents.

* **Stage 2: Dense Sentence Retrieval**
    * Processes sentences from the documents selected in Stage 1.
    * Two approaches were explored for encoding claims and potential evidence sentences:
        * **DPR-based:** Uses separate encoders (DPRContextEncoder for evidence, DPRQuestionEncoder for claims) derived from BERT. The similarity between claim and evidence embeddings determines relevance.
        * **Single bert-base Encoder:** Uses a single bert-base-uncased model to encode both claim and evidence (formatted as `[CLS] evidence [SEP] claim`). Cosine similarity between these embeddings is used for ranking.
    * The top few (e.g., five) most relevant sentences are passed to the NLI module.

### 2. Natural Language Inference (NLI) Module

* Takes the claim and the retrieved evidence sentences as input.
* Predicts whether the evidence supports the claim, refutes it, or if there's not enough information.
* Two main model architectures were explored:
    * **GPT2ForSequenceClassification:** A decoder-only transformer model (GPT-2) adapted for sequence classification. Claims and evidence are concatenated and fed to the model.
    * **BertForSequenceClassification:** An encoder-only transformer model (bert-base-uncased) adapted for sequence classification. The input is formatted as `[CLS] claim [SEP] evidence_concatenated`.

## Findings

* Incorporating Named Entities (NER) in the sparse retrieval stage significantly improved the Mean Reciprocal Rank (MRR) of finding relevant documents (e.g., MRR@1 improved from ~0.0009 to ~0.5342).
* This improvement in retrieval led to better overall system performance:
    * Label Accuracy increased by over 10% (to ~0.6053).
    * Evidence F1 score increased by over 15% (to ~0.2101).
    * The strict FEVER score improved significantly (from ~0.0811 to ~0.3891).
* The NLI model sometimes showed confusion with 'NOT ENOUGH INFO' claims, potentially due to patterns learned from the retrieved (but not always perfectly relevant) evidence for this category.
