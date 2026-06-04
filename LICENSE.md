# SFM Research Preview License — v1.0-preview

Released by **Sai T. Reddy** (ETH Zürich / Botnar Institute of Immune Engineering) and co-authors as an academic research release accompanying the SFM preprints (2026).

**STATUS — RESEARCH PREVIEW.** These terms grant free research use of the materials below. Commercial-use and patent-licensing terms are **not** part of this preview; they are being arranged with the institutional rights-holders and will be published separately.

**1. Materials covered.** "Materials" means, for each Specificity Foundation Model (SFM) the authors release — including dtSFM — (a) the source code ("Software"), (b) the trained parameters ("Weights") and parameters derived from them ("Derivatives"), and (c) data produced by running the model ("Outputs": predicted scores, generated SMILES, candidate sequences, embeddings).

**2. Research-use grant.** The authors grant you a worldwide, royalty-free, non-exclusive license to use, reproduce, modify, and redistribute the Materials for **Research Use**, at no cost and without contacting the authors. Research Use includes academic, educational, non-profit, government, **and industry research** — internal evaluation, characterization, screening triage, benchmarking, methodology development, and pipeline prioritization by companies — together with publication of results and open distribution of Derivatives, subject to §3–§8.

**3. Published molecules are free for everyone.** The specific molecular Outputs disclosed in the accompanying preprints — the generated variants, SMILES strings, and sequences reported in those papers — are **dedicated to the public**. Anyone may make, use, and commercialize those disclosed molecules for any purpose, with no license and no obligation to the authors or their institutions.

**4. Patents reserved; no patent license granted.** The SFM architectures and training methods are the subject of **pending patent applications** (see the Patents Notice below). These terms are a **copyright permission for Research Use only** and grant **no license, express or implied, under those patent applications**. Nothing here authorizes use of the model or its methods to discover, design, optimize, or select candidates that are then developed into a commercial product or service.

**5. Commercial use — deferred.** Commercial use of the Materials (other than the molecules dedicated under §3) is outside this preview. Commercial-licensing and patent terms are being arranged with the institutional rights-holders (ETH Zürich and the Botnar Institute of Immune Engineering) and will be published separately. Parties interested in commercial use may contact **sai.reddy@ethz.ch** for early discussion; no commercial rights are granted until those terms are issued.

**6. Prohibited uses.** You may not use the Materials to develop, design, or produce biological, chemical, or nuclear weapons or their means of delivery, or for any use that violates applicable law.

**7. Attribution.** If you use or redistribute the Materials, retain this notice and cite: Reddy (2026), *Methods for Molecular Recognition Computing*; Reddy (2026), *Vibe-Coding Specificity Foundation Models*; and the relevant domain SFM paper(s) (e.g., the dtSFM papers). Include these terms with any redistributed Derivative.

**8. No warranty; no liability.** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. The authors and their institutions make no warranty that Outputs are accurate, reliable, or safe and accept no liability for any use of the Materials or Outputs. Any clinical, product, or downstream application is the sole responsibility of the party undertaking it and requires independent scientific, clinical, and regulatory validation.

---

## Patents Notice *(informational)*

Twelve (12) patent applications covering the Specificity Foundation Model (SFM) architecture and training methods have been **filed**. **Sai T. Reddy is the sole inventor on ten of the twelve applications, including the drug–target SFM (dtSFM), and the primary inventor on all twelve.** The applications are held by ETH Zürich and the Botnar Institute of Immune Engineering (BIIE).

**The applications cover:** the Specificity Foundation Model architecture (dual sequence encoders, symmetric contrastive learning, and cross-attention); **the method of using a three-dimensional structural-prediction model (such as AlphaFold) to generate the training labels or supervision for a sequence-based binding or specificity model** — that is, training an SFM, or any sequence-native interaction model, on labels derived from a structural model; and per-domain SFM implementations (drug–target, antibody–antigen, TF–DNA, peptide–MHC, CRISPR gRNA–DNA, miRNA–mRNA, enzyme–substrate, TCR–pMHC, RBP–RNA).

**They do NOT cover:** the molecules, sequences, and other Outputs disclosed in the accompanying papers (dedicated to the public — free for any use, including commercial); standard ML components (transformer attention, contrastive loss, PyTorch); or third-party pretrained backbones (ESM-2, MoLFormer-XL, DNABERT-2, …) and training-data sources, which remain under their own licenses.

**Effect on you:** research use of the models and code is free under this license and is unaffected by these applications. Commercial-licensing and patent terms are being arranged with the institutional rights-holders and will be published separately; contact **sai.reddy@ethz.ch** for early discussion. Application numbers, jurisdictions, and status are available on request.

---

*Released by Sai T. Reddy · sai.reddy@ethz.ch · 2026 · SFM Research Preview License v1.0-preview.*
*On public release this file becomes `LICENSE.md` at the repository root and is attached to the Hugging Face model card.*
