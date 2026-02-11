# Test PSOFT as a New PEFT Tuner

This repository contains the implementation and evaluation setup for integrating PSOFT as a new PEFT tuner.

The implementation is based on `peft-v0.17.0` from https://github.com/fei407/PSOFT, extended with the newly developed PSOFT tuner modules.

We further evaluated the implemented PSOFT tuner on real-world datasets, following the same experimental settings as described in the paper. Specifically, experiments were conducted on the CoLA dataset using DeBERTa-v3-base and on MetaMathQA-40K using Llama-3.2-3B, with evaluation on the GSM8K test set. The obtained results are consistent with those reported in the paper.
