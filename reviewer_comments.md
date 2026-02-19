# Reviewer Comments

**Comments:** (Reviewer attachments will be available in your author page)

---

## Reviewer #2

This manuscript develops ML models (including an ensemble/stacking approach) to predict glass transition temperature (Tg) of lignin-based polyurethane materials using formulation and processing descriptors. The topic is timely and potentially useful for formulation screening. However, there are several methodological and framing issues that currently limit confidence in the reported performance and the "design/discovery" claims. I list major concerns and actionable revisions below.

### Major Concerns

1. **Stacking Procedure and Data Leakage**  
   The described stacking procedure appears to generate meta-features from base-model predictions on the full dataset, then trains the meta-model on those predictions. This possible leak training information into the meta-model and produce overly optimistic validation metrics. Re-implement stacking using out-of-fold (OOF) predictions to build meta-features (e.g., StackingRegressor with internal CV or cross_val_predict), and evaluate with nested CV or a strict held-out test set. Please clearly state whether each reported scatter/correlation plot is based on OOF/test predictions rather than in-sample fitted values.

2. **Swelling Ratio as Input Feature**  
   Including swelling ratio as an input feature can greatly improve prediction but changes the task from "predict Tg from recipe/controllable variables" to "predict Tg using a post-synthesis characterization measurement." This weakens the manuscript's formulation-design narrative.

3. **Deterministically Linked Inputs**  
   Several inputs appear to be deterministically linked (e.g., ratio vs mmol NCO vs wt% NCO; complementary composition variables), which can destabilize coefficients/feature importance and create redundant search dimensions in formulation mapping.

4. **Overfitting Concerns**  
   Large train-validation gaps for certain models indicate overfitting, raising concerns about generalization. High reported correlations (e.g., Pearson ~0.99) may reflect in-sample fits rather than true predictive power.

5. **Mechanistic Discussion**  
   Strengthen the mechanistic discussion with at least one experimentally grounded link supporting why the model's learned relationships are physically meaningful.

---

## Reviewer #3

This manuscript (MADE-D-25-05345) proposed a machine learning ensemble model to predict the glass‑transition temperature (Tg) of lignin‑based polyurethanes (lignin-PUs) using a limited experimental dataset. The proposed stacking ensemble achieves a mean absolute error of 6.66 °C on the validation set, and the development of a user‑friendly interface for formulation exploration represents a tangible step toward practical application. However, the manuscript in current form falls short of the expected conceptual depth and broader scientific impact required for publication in a journal focusing on materials design. The main weaknesses are the inadequate justification for choosing lignin‑based polyurethanes as the model system, and the tenuous link between the model outputs and polymer physical chemistry principles. The authors need improve significantly the manuscript to meet the requirements of Materials & Design.

### Major Issues

1. **Lignin Variability**  
   As a heterogeneous polymer, lignin exhibits significant variations in molecular weight and basic structural units among different sources. How can we ensure precise prediction of Tg of lignin-PUs?

2. **Justification for Research Subject**  
   The introduction did not make it clear why the authors selected lignin-PUs as the research subject for predicting Tg. Authors should thoroughly justify the necessity and significance of studying Tg of lignin-PUs in the Introduction section.

3. **Generalizability to Different Lignin Sources**  
   All experiments are based on the same batch of extracted Kraft lignin, which raises the following question: Is the model only effective for this specific lignin batch? If the lignin source is changed (usually accompanied by notable variations in structural parameters such as molecular weight, composition of basic structural units, and functional groups), would the model require retraining or adjustment to maintain prediction accuracy?

4. **Design and Optimization Guidance**  
   For materials design, the developed Tg prediction model for lignin-PUs should inform and guide the rational design and optimization of polyurethane architectures. It is recommended to add a relevant summary in the structural discussion section.

5. **Connection to Polymer Physics**  
   The interpretation of the model results should be more deeply connected to established polymer physics theories. It is recommended to explicitly discuss how the key predictive features identified by the model correlate with fundamental concepts such as crosslink density, chain mobility, and free volume theory to explain their influence on the glass transition temperature.

6. **Swelling Ratio Issue**  
   The manuscript incorporates the "swelling ratio" as one of the input features and highlights its importance. However, the swelling ratio is a post-synthesis material property characterization, not a formulation parameter. In practical applications, predicting the Tg value of a new formulation would synthesize the material firstly, which appears to contradict the intended "predict-then-design" workflow.

7. **Minor Issues**
   - **(1) Terminological consistency:** Co-polyol type or Co-polyol Type, Lignin-based polyurethanes or lignin PU, lignin-PUs, not just these.
   - **(2) Improving the clarity of the Figures.**

---

## Reviewer #5

The manuscript discusses ensemble (ML) models to predict Tg from PU-related input parameters. Unfortunately, the paper needs major corrections, as discussed below.

### Major Corrections Required

1. **Training Error vs Test Error**  
   The authors report a "training" error instead of a "test" error. Hyperparameters were optimized using the whole dataset (via 5-fold CV) and then the ensemble model was optimized again using the whole dataset (actually, the data splitting strategy used in the ensemble optimization is poorly described). Finally, the best, optimized ensemble model was tested using the whole dataset again (!), obviously leading to small MAE or MSE errors, as expected for training errors. The authors would need to apply the final ensemble model on an unseen test data that was never used in any moment by the hyperparameter optimization or ensemble optimization steps. Training errors are not valid as model evaluation, so that many results described in this paper are not physically meaningful. Authors should apply a nested CV approach (e.g., 5-fold CV is performed inside the training folds of the main 5-fold CV) to check the test error and generalizability of the model.

2. **Overfitting Discussion**  
   Because the test error was never calculated, but only the training error (of 6.66 °C), overfitting could never be discussed, which was one of the negative points found in the discussed literature in the state-of-the-art section. Also, because no test error was calculated, the generalizability of the model, which is among the most important metrics for any ML model, cannot be discussed at all.

3. **Swelling Ratio as Input Parameter**  
   In the current manuscript, using an output property (swelling ratio) as input parameter in the ML models simply prevents using the final trained model for any kind of useful predictions that could be verified experimentally. E.g., a set of promising parameters is theoretically chosen, which should lead to a desired Tg value. Theoretically, one can very easily suggest new, promising input parameters (including the swelling ratio) whose experiment should be performed afterwards. However, the corresponding experiment is then impossible to be performed because one of the "input" parameters (the swelling ratio) is actually a final output parameter which cannot be set in advance in the chosen experiment: it is a consequence of the experiment itself. Of course the authors could use just the input parameters (without the swelling ratio) to predict Tg, which would then bring back the usefulness of the model, but these inputs are not informative enough (or the authors didn't find or select good input parameters for their model), as already reported in the manuscript.

4. **State-of-the-Art Comparison**  
   In the state-of-the-art, the authors state that the GBR model of ref 19 was strongly overfitting, which is misleading, as the best screened ML model in that paper was the LASSO model, which exhibited very little overfitting, as extensively discussed in that paper and also highlighted in Fig. 7. The GBR model reported in ref 19 was actually the worst screened model: one does not draw any conclusions from the worst ML model screened, but from the best ML model screened. In addition, the MAE error reported for the LASSO model used in Ref 19, as correctly evaluated on the test set, was between 4-5 °C, using only as few as 35 samples, being therefore substantially better than the ensemble model of the present manuscript, which used as many as 136 samples.

5. **Extrapolation Performance**  
   Testing the extrapolation performance of the model would make more sense if a gaussian processes regressor would have been included in the screened ML models.

---

## Reviewer #6

A well done timely manuscript.
