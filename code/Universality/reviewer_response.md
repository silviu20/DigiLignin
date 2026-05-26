# Response to Reviewer #3

---

**Reviewer comment:**
> The revised manuscript strengthens the necessity and scientific value of predicting the Tg of lignin-based polyurethanes, and addresses the reviewers' concerns regarding the potential impact of lignin heterogeneity on prediction accuracy. The quality of the manuscript has been greatly improved. However, it should be noted that the universality and accuracy of the Tg prediction model for lignin-based polyurethanes still require further validation.

---

**Response:**

We thank the reviewer for the positive assessment of the revised manuscript and for this constructive remark. We agree that explicitly addressing the model's scope of validity strengthens the scientific rigor of the work, and we have performed two additional analyses to directly respond to this concern: (i) a leverage-based applicability domain (AD) analysis and (ii) a permutation test. Both have been added to the manuscript as a new subsection (Section 3.6, *Applicability Domain Analysis*) with two new figures.

**1. Applicability domain — addressing "universality"**

We performed a leverage-based AD analysis using the Williams plot, which is the standard framework for defining the chemical space within which a QSPR model produces reliable predictions. Leverage values (*h*ᵢ) were computed from the HAT matrix of the scaled training set, and standardised residuals (*z*ᵢ = *e*ᵢ/*σ*) were derived using the training residual standard deviation. The warning threshold was set at *h*\* = 3(*k* + 1)/*n* = 0.173 (*k* = 5 features, *n* = 104 training samples). A sample was considered within the AD if it simultaneously satisfied *h*ᵢ ≤ *h*\* and |*z*ᵢ| ≤ 3.

The results show that 98.1% of training samples lie within the AD, confirming internal calibration. For the validation and test sets, 68.8% (11/16) and 62.5% (10/16) of samples fall within the AD, respectively. Samples outside the AD are characterised by atypically high leverage — reflecting structural dissimilarity from the training centroid — or by large standardised residuals. This finding directly contextualises the reported prediction errors: within the defined chemical space (Kraft EKL, PTHF co-polyols of Mₙ = 250–1000 g/mol, HDI/HDIt isocyanates, [NCO]/[OH] = 0.6–1.4), the model produces reliable predictions. Formulations outside this space carry increased uncertainty, and we have made this limitation explicit in the manuscript. We note that this is not a deficiency unique to our model: the same leverage-based AD framework has been applied in published QSPR studies on polymers and thermosets precisely because it provides a principled and transparent boundary for model applicability (see, e.g., Gramatica, *QSAR Comb. Sci.* 2007, 26, 694).

The AD analysis also explains why the reviewer's concern about "universality" is correctly framed as a *scope* question rather than a *failure* question: the model is valid and reliable within its training domain, and the AD boundary makes that domain explicit for any user attempting to apply the model to a new formulation.

**2. Statistical validation — addressing "accuracy"**

To confirm that the model has captured genuine structure–property relationships rather than spurious correlations inherent to a small dataset, we performed a permutation test (1000 permutations). The Tg labels of the training set were randomly shuffled in each permutation, and the validation MAE was re-evaluated while preserving the fitted base models. The true validation MAE of 13.41 °C lies substantially below the fifth percentile of the permuted distribution (22.02 °C), corresponding to *p* < 0.001. This result provides rigorous statistical confirmation that the predictive performance of the stacking ensemble is not attributable to chance, offering additional evidence for the validity of the model within its defined applicability domain.

**Changes to the manuscript:**

- A new subsection, *3.6 Applicability Domain Analysis*, has been added immediately after Section 3.5 (*Evaluation of Model Fit and Residual Distribution*). The text describes the Williams plot methodology, the leverage threshold derivation, and the AD membership results for all three data partitions.
- **Figure 8** (two-panel) has been added: panel (A) is the Williams plot; panel (B) is the AD coverage bar chart showing the percentage of validation and test samples inside/outside the AD.
- The permutation test result (*p* < 0.001) is reported in the text of Section 3.6. The corresponding histogram (Figure S[X]) has been added to the Supporting Information.
- The existing Figures 8–10 (KDE mapping plots, extrapolation, parallel coordinates) have been renumbered to Figures 9–11 accordingly.

---

*All analysis scripts and output files (williams_plot, ad_coverage, feature_coverage, permutation_test) are available in the* `Universality/` *folder of the project repository.*
