# Test PSOFT as a new PEFT tuner
This repository contains the implementation and testing setup for integrating PSOFT as a new PEFT tuner.

The implementation is based on peft-v0.17.0 from https://github.com/fei407/PSOFT, extended with new PSOFT tuner codes.

We evaluated the new PSOFT tuner on the CoLA dataset using DeBERTa-v3-base and on MetaMathQA-40K using Llama-3.2-3B, with evaluation on the GSM8K test set. The obtained results are consistent with those reported in the paper.
