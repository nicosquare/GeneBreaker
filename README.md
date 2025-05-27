# GeneBreaker: Systematic Jailbreak Evaluation for DNA Foundation Models

## Overview

![GeneBreaker Overview](assets/GeneBreaker.png)

## Introduction

GeneBreaker is the first framework to systematically evaluate jailbreak vulnerabilities in DNA foundation models. As DNA foundation models become increasingly capable of designing synthetic functional DNA sequences, understanding and mitigating their potential to generate harmful, pathogenic, or toxin-producing genes is critical for biosecurity.

### Paper Abstract

DNA, encoding the genetic instructions for all living organisms, fuels groundbreaking advancements in genomics and synthetic biology. Recently, DNA Foundation Models have achieved the success of designing synthetic functional DNA sequences, but their susceptibility to jailbreaking remains underexplored, leading to potential concern of generating harmful sequences such as pathogenic or toxin-producing genes. In this paper, we introduce GeneBreaker, the first framework to systematically evaluate jailbreak vulnerabilities in DNA foundation models. GeneBreaker employs (1) an LLM agent with customized bioinformatics prompts to design high-homology, non-pathogenic prompts, (2) a beam search guided by PathoLM and log-probability heuristics to steer generation toward pathogen-like sequences, and (3) a BLAST-based evaluation pipeline against a curated Human Pathogen Database (JailbreakDNABench) to detect successful jailbreaks.

Evaluated on our curated JailbreakDNABench, GeneBreaker consistently successfully jailbreaks the latest Evo series models across 6 viral categories (up to 60% Attack Success Rate for Evo2-40B). Further case studies on SARS-CoV-2 spike protein and HIV-1 envelope protein demonstrate the sequence and structural fidelity of jailbreak outputs, while evolutionary modeling of SARS-CoV-2 underscores biosecurity risks. Our findings also reveal that scaling DNA foundation models amplifies dual-use risks, motivating enhanced safety alignment and tracing mechanisms.

---

## GeneBreaker Pipeline

GeneBreaker consists of the following key components:

1. **LLM Agent with Bioinformatics Prompts**
   - Designs high-homology, non-pathogenic prompts to initiate model generation.
2. **Beam Search Guided by PathoLM and Log-Probability Heuristics**
   - Steers the DNA foundation model to generate sequences with high similarity to known pathogens.
3. **BLAST-based Evaluation Pipeline**
   - Compares generated sequences against the curated Human Pathogen Database (JailbreakDNABench) to detect successful jailbreaks.

---

## Benchmark: JailbreakDNABench

![JailbreakDNABench Benchmark](assets/JailbreakDNABench.png)

---

## Usage Example: Jailbreaking Evo2 to Generate HIV-like Sequences

The script `auto_jailbreak_hiv.py` automates the process of jailbreaking Evo2 to generate high-similarity sequences to HIV.

### Run the Jailbreak Script

```bash
python auto_jailbreak_hiv.py --skip_chatgpt
```

This script will:
- Use the GeneBreaker pipeline to prompt Evo2 for sequence generation.
- Guide the generation toward HIV-1 envelope protein-like sequences.
- Evaluate the generated sequences for similarity to known HIV sequences (achieving up to 92.50% DNA similarity and 87.79% protein similarity).

---

## Case Study

![Case Study](assets/case.png)

---

For more details, see the code in this directory and the associated paper. For questions or contributions, please contact the authors. 