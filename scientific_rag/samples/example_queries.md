# Scientific RAG Pipeline — Sample Query Results

Ten representative questions answered by the pipeline, showing retrieval sources and generated answers.

**Index:** 447 chunks across 6 biomedical PDFs  
**Embedder:** PubMedBERT (768 dims)  
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2  

---

## Q1. What role does TAF6 delta play in inducing apoptosis?

**Chunks retrieved:** 3  
**Sources:**
- `orchestrates an apoptotic transcriptome` — Background (score: 1.0, via: hyde+bm25+reranked+dense)
- `Theoretical Biology and Medical` — Circumstantial context analysis on actual transcriptome (score: 0.316, via: hyde+bm25+reranked+dense)

**Answer:**

TAF6 δ plays a pro-apoptotic role by inducing programmed cell death independently of p53 [orchestrates an apoptotic transcriptome — Background]. It acts downstream and independently of p53 to control gene expression at the onset of apoptosis [8]. The use of splice-switching oligonucleotides (SSO) to experimentally direct the expression of endogenous TAF6 δ in living cells has demonstrated its pro-apoptotic activity [5]. 

References:
- [orchestrates an apoptotic transcriptome — Background] 
- [8] (presumably the same paper as "orchestrates an apoptotic transcriptome")
- [5] (cited in "orchestrates an apoptotic transcriptome — Background")

---

## Q2. How does inhibition of mitochondrial respiration affect BRAF mutant melanoma brain metastasis?

**Chunks retrieved:** 10  
**Sources:**
- `Inhibition of mitochondrial respiration` — Discussion (score: 1.0, via: hyde+bm25+reranked+dense)
- `UC Davis Previously Published Works` — Results (score: 0.683, via: hyde+bm25+reranked+dense)

**Answer:**

Inhibition of mitochondrial respiration prevents resistance to BRAF inhibition (BRAFi) therapy in BRAF-mutant melanoma brain metastasis [Inhibition of mitochondrial respiration — Discussion]. Specifically, β-sitosterol, which inhibits mitochondrial respiration, efficiently prevented resistance to BRAFi therapy in vivo in BRAF-mutant melanoma brain metastasis [Inhibition of mitochondrial respiration — Discussion]. This is consistent with the observation that BRAF-mutant melanomas treated with vemurafenib (a BRAFi) exhibit up-regulated mitochondrial respiration, leading to potential resistance [Inhibition of mitochondrial respiration — Results]. 

Therefore, inhibiting mitochondrial respiration, such as with β-sitosterol, may be a useful strategy to prevent resistance to BRAFi therapy in BRAF-mutant melanoma brain metastasis.

---

## Q3. What is the function of ParB protein in centromere biology?

**Chunks retrieved:** 15  
**Sources:**
- `Permissive zones for the centromere-binding protein` — INTRODUCTION (score: 1.0, via: hyde+bm25+reranked+dense)

**Answer:**

ParB is a DNA-binding protein that nucleates on a parS sequence, which is a centromere-like DNA sequence [Permissive zones for the centromere-binding protei]. It is involved in centromere biology by binding to parS and spreading to surrounding regions, up to a maximum of 2 kb around individual parS [Permissive zones for the centromere-binding protei]. The protein is essential for its function, as demonstrated by the viability of Caulobacter cells producing FLAG-tagged ParB, indicating that the tag does not interfere with ParB function [Permissive zones for the centromere-binding protei — Caption: Figure 1]. 

The exact mechanism and extent of ParB spreading may vary, but its role in binding to parS is a crucial aspect of its function. [Permissive zones for the centromere-binding protei].

---

## Q4. What antisense oligonucleotides were used to induce TAF6 delta expression?

**Chunks retrieved:** 15  
**Sources:**
- `orchestrates an apoptotic transcriptome` — Methods (score: 1.0, via: hyde+bm25+reranked+dense)
- `Controls Apoptosis and Gene Expression in the` — Caption: Figure 2 (score: 0.968, via: hyde+bm25+reranked+dense)

**Answer:**

The antisense oligonucleotides used to induce TAF6 delta expression were SSO T6-1 and SSO T6-3, as well as Taf6 AS1. 

Specifically, SSO T6-1 and SSO T6-3 were used to force splicing from the distal 5' SS and enforce expression of the endogenous TAF6 δ isoform [Controls Apoptosis and Gene Expression in the — TAF6 d protein levels]. Taf6 AS1 was also shown to induce TAF6 δ mRNA expression and concurrently decrease the level of the major TAF6 α mRNA [Controls Apoptosis and Gene Expression in the — Selective induction of endogenous TAF6 d mRNA]. 

Additionally, 2'-O-methyl-oligoribonucleoside phosphorothioate antisense 20-mers from Sigma-Proligo were used as the antisense oligonucleotides [Controls Apoptosis and Gene Expression in the — Selective induction of endogenous TAF6 d mRNA].

---

## Q5. How was cell viability measured in the mitochondrial inhibition experiments?

**Chunks retrieved:** 8  
**Sources:**
- `Inhibition of mitochondrial respiration` — Results (score: 1.0, via: bm25+reranked+dense)

**Answer:**

The context does not directly state how cell viability was measured. However, it mentions "Long-term colony formation assays" [Inhibition of mitochondrial respiration — Results], which suggests that cell viability or growth was assessed through colony formation assays. 

Therefore, the answer is that cell viability was measured using long-term colony formation assays [Inhibition of mitochondrial respiration]. Context is partially insufficient to provide a definitive answer.

---

## Q6. How do TAF6 delta induction and BRAF inhibition both affect tumor cell survival, and what mechanisms do they share?

**Chunks retrieved:** 15  
**Sources:**
- `orchestrates an apoptotic transcriptome` — Background (score: 1.0, via: hyde+bm25+reranked+dense)
- `Controls Apoptosis and Gene Expression in the` — Discussion (score: 0.928, via: hyde+reranked+dense)
- `Theoretical Biology and Medical` — Circumstantial context analysis on actual transcriptome (score: 0.173, via: hyde+reranked+dense)

**Answer:**

The context is insufficient to directly compare the effects of TAF6 delta induction and BRAF inhibition on tumor cell survival, as there is no mention of BRAF inhibition in the provided context. However, it is stated that "targeting TAF6 results in a substantially more robust apoptotic response than targeting another apoptotic gene, Bcl-x in several tumor cell lines" [orchestrates an apoptotic transcriptome — Discussion]. 

TAF6 δ induces apoptosis in cell lines that lack p53 expression, and the induction of TAF6 δ produced similar levels of apoptosis in the HCT-116 p53 -/- colon carcinoma cell line as in its p53 positive counterpart [orchestrates an apoptotic transcriptome — Discussion]. The mechanisms by which TAF6 δ promotes apoptosis are not fully understood, but it is known that TAF6 δ interacts with p53 in vitro [orchestrates an apoptotic transcriptome — Discussion]. 

Further characterization of the TAF6 δ signaling hub may provide novel therapeutic avenues to induce controlled tumor cell death irrespective of their p53 status [Controls Apoptosis and Gene Expression in the — Discussion] and [orchestrates an apoptotic transcriptome — Discussion].

---

## Q7. Which studies show that apoptosis can be triggered independently of p53, and how do their mechanisms differ?

**Chunks retrieved:** 15  
**Sources:**
- `Controls Apoptosis and Gene Expression in the` — TAF6 d activates gene expression independently of p53 (score: 1.0, via: hyde+bm25+reranked+dense)
- `orchestrates an apoptotic transcriptome` — Background (score: 0.8, via: hyde+bm25+reranked+dense)

**Answer:**

According to the provided context chunks from the paper "Controls Apoptosis and Gene Expression in the — TAF6 d activates gene expression independently of p53", TAF6 d can induce apoptosis independently of p53. The study shows that TAF6 d induces apoptosis in cells lacking p53 and activates gene expression independently of cellular p53 status [Controls Apoptosis and Gene Expression in the — TAF6 d activates gene expression independently of p53]. 

The mechanism involves the activation of genes such as HOM-TES-103, HES1, and ACRC independently of p53, which are candidate genes that could mediate TAF6 d-dependent apoptosis [Controls Apoptosis and Gene Expression in the — Discussion]. The TAF6 δ pathway represents a distinct mechanism that controls apoptosis without p53 but has interconnections to the p53 pathway, including shared target genes [Controls Apoptosis and Gene Expression in the — Discussion].

Since there is only one paper referenced in the context, there are no differing mechanisms from multiple studies to compare. The context is limited to a single study, and therefore, it is insufficient to compare mechanisms across different studies.

---

## Q8. What quantitative effect on apoptosis was demonstrated when TAF6 delta was expressed?

**Chunks retrieved:** 14  
**Sources:**
- `Controls Apoptosis and Gene Expression in the` — Caption: Figure 6 (score: 1.0, via: hyde+bm25+reranked+dense)
- `Theoretical Biology and Medical` — Transcriptome data (score: 0.973, via: hyde+bm25+reranked)
- `orchestrates an apoptotic transcriptome` — Abstract (score: 0.966, via: hyde+bm25+reranked+dense)

**Answer:**

Endogenous TAF6δ expression induces apoptosis in the absence of p53 [TAF6 d Controls Death Sans p53 PLoS ONE]. Specifically, it was demonstrated that treatment of HeLa cells with a specific oligonucleotide resulted in a significant increase in apoptotic cells, as indicated by a 3.5-fold increase in KRT18c positive cells [Controls Apoptosis and Gene Expression in the — Endogenous TAF6 d expression causes apoptosis in HeLa]. The pro-apoptotic activity of TAF6δ has been demonstrated using modified antisense RNA oligonucleotides to experimentally direct its expression in living cells [Controls Apoptosis and Gene Expression in the — TAF6 d induces apoptosis in the absence of p53].

---

## Q9. What evidence across the papers supports the link between gene expression reprogramming and apoptosis?

**Chunks retrieved:** 3  
**Sources:**
- `Controls Apoptosis and Gene Expression in the` — Introduction (score: 1.0, via: hyde+bm25+reranked+dense)
- `orchestrates an apoptotic transcriptome` — Caption: Figure 6 (score: 0.211, via: hyde+bm25+reranked)

**Answer:**

The evidence across the papers supports that gene expression plays a crucial role in apoptosis. Specifically, it is stated that "Gene expression patterns of pro- and anti-apoptotic genes... can dictate the life-or-death decisions of cells" [Controls Apoptosis and Gene Expression in the — Introduction]. Furthermore, the levels of numerous death-related genes can be induced during apoptosis [Controls Apoptosis and Gene Expression in the — Introduction]. 

The papers also provide evidence that specific proteins and complexes, such as p53 and TAF6, are involved in controlling apoptosis by altering gene expression. For example, "The most intensely studied protein known to control apoptosis by altering gene expression is the p53 tumor suppressor" [Controls Apoptosis and Gene Expression in the — Introduction]. Moreover, TAF6δ can induce apoptosis in the absence of p53, as shown in Figure 6 of "Controls Apoptosis and Gene Expression in the — Caption: Figure 6" and "orchestrates an apoptotic transcriptome — Caption: Figure 6". The latter figure also provides evidence for transcriptional crosstalk between endogenous TAF6δ and p53.

These findings collectively suggest a link between gene expression reprogramming and apoptosis, with key roles played by proteins such as p53 and TAF6δ [Controls Apoptosis and Gene Expression in the — Introduction; orchestrates an apoptotic transcriptome — Caption: Figure 6].

---

## Q10. What are the key transcription factors involved in regulating apoptosis pathways across these studies?

**Chunks retrieved:** 3  
**Sources:**
- `orchestrates an apoptotic transcriptome` — Background (score: 1.0, via: hyde+bm25+reranked+dense)
- `Theoretical Biology and Medical` — Circumstantial context analysis on actual transcriptome (score: 0.47, via: hyde+bm25+reranked+dense)
- `Controls Apoptosis and Gene Expression in the` — Caption: Figure 5 (score: 0.222, via: hyde+bm25+reranked+dense)

**Answer:**

Based on the provided context, the key transcription factors involved in regulating apoptosis pathways are:

1. TAF6δ (also known as TAF6 delta), an inducible splice variant of the TFIID subunit TAF6, which controls cell death through the TAF6δ pathway [1: "orchestrates an apoptotic transcriptome — Background"].
2. p53, a gene that is often involved in regulating apoptosis, but is shown to be dispensable for TAF6δ-induced apoptosis [3: "Controls Apoptosis and Gene Expression in the — Caption: Figure 5"].

It is demonstrated that TAF6δ acts downstream and independently of p53 to control gene expression at the onset of apoptosis [2: "Theoretical Biology and Medical — Circumstantial context analysis on actual transcriptome"]. 

Therefore, TAF6δ is a key transcription factor involved in regulating apoptosis pathways.

---
