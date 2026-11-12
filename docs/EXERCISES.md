# Optional exercises: what you should find

Run steps ① to ④ of `NEBULA_Lab.ipynb` first, then the cells under "Optional: explore after class". The values below come from the executed notebook with erythroid precursors held out, and yours will match them exactly.

### 1. Where does chromatin beat the control?
The copy-the-average control wins overall, but it cannot see a promoter that opens only in the held-out cell type. The exercise takes the genes whose promoter opens most in erythroid precursors compared with their average in the training cell types, picked from the inputs alone. The top 8 are IFIT1B, HBB, TRIM10, TMEM86B, MRGPRE, SEC14L4, ART4 and RHAG, several of them classic red-cell genes. The accessibility model is closer to the measurement than the control for 6 of these 8, and for 41 of the top 50. The two misses, TMEM86B and MRGPRE, show that an opening promoter does not guarantee RNA.

### 2. Is erythroid typical?
Hiding B cells instead gives A 0.626, B 0.625 and control 0.902. The accessibility models land in the same place, while the control does much better, because B cells' expression is closer to the average of the other 15 cell types than erythroid precursors' is.

### 3. Does option B ever beat option A by a meaningful margin?
No. Across all 16 hold-outs, A ranges from 0.51 to 0.63, B minus A from −0.005 to +0.003, and the control from 0.77 to 0.97. Erythroid precursors are the control's hardest case.

### 4. Would looking further from the gene help?
Promoter only gives 0.621; adding the 10, 50, 100 or 250 kb windows gives 0.616 to 0.618, and all four together 0.616. Summed openness, at any distance, adds nothing a straight line can use.

### 5. Is the straight line the problem?
Gradient-boosted trees give 0.625 with the promoter, 0.625 with promoter plus 50 kb, and 0.643 with all four windows. Flexibility helps a little, and only with several distances at once. GET reads *which* regions are open and which motifs they carry, information that a sum of openness throws away.

### 6. Is a high r about knowing the gene, or knowing the cell type?

| Model | Overall r | Cell-type-specific r |
|---|---|---|
| Accessibility model (step 3) | 0.621 | 0.41 |
| Gene's usual level only | 0.773 | 0.00 |
| Usual level + promoter + its change | 0.812 | 0.41 |

The cell-type-specific r compares the model's change for this cell type (its prediction minus its own average prediction for that gene) with the measured change (measured minus the gene's usual level). The usual level wins overall but carries no cell-type-specific information, because its prediction is the same in every cell type. All of the cell-type-specific signal comes from chromatin, and combining the two gives the best overall score, 0.81.

### 7. Pearson's r by hand
The hand calculation and NumPy both give 0.6211: r is the covariance of the two variables divided by the product of their spreads.

### 8. The biggest misses
The largest under-predictions are erythroid identity genes; HBB is about 1,600-fold above its prediction. The largest over-predictions are open promoters with almost no RNA: an open promoter makes transcription possible, and whether it happens depends on more than openness.

### 9. Why score only open promoters?
The same model scores 0.621 on the 14,826 open-promoter genes and 0.641 when the 3,459 closed-promoter genes are added. Closed, silent genes are easy to predict, so including them flatters the model. A reported r always depends on which genes it was computed on.

### 10. Why pretraining matters (published values)
In Extended Data Fig. 2d, GET trained without its pretraining stage scores 0.60 to 0.61, below the 0.78 copy-the-average baseline, while the pretrained and fine-tuned model scores 0.94. The notebook redraws these values; it does not compute them.

## Discussion questions
1. Option B did not help. Does that mean enhancers do not regulate genes? (No. Summed openness within 50 kb carries no extra information for this model. Regulation is tested by perturbation, for example CRISPRi of the element.)
2. The control scores 0.77 without looking at chromatin. What does that say about a reported r of 0.94?
3. GET's astrocytes were held out of fine-tuning but not of pretraining. Ours were held out of all model training, although their chromatin helped define the shared list of open regions. Which test is harder, and why might it matter?
