$ ./run_step_by_step.sh
================================================================================
STEP-BY-STEP ANALYSIS EXECUTION
================================================================================

STEP 1/3: Running VIF Analysis...
Traceback (most recent call last):
  File "<string>", line 36, in <module>
    recommendations = recommend_feature_reduction(vif_df, threshold=10)
  File "C:\Users\sacaru\digilignin\DigiLignin\5.Model\VIF_Analysis_Multicollinearity.py", line 158, in recommend_feature_reduction
    print(f"\n2. MODERATE multicollinearity (5 < VIF \u2264 {threshold}): CONSIDER REMOVING")
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sacaru\AppData\Local\Python\pythoncore-3.14-64\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2264' in position 41: character maps to <undefined>
Data loaded: 136 samples

================================================================================
VARIANCE INFLATION FACTOR (VIF) ANALYSIS
================================================================================

Calculating VIF for features:
  - Lignin (wt%)
  - Co-polyol (wt%)
  - Co-polyol type (PTHF)
  - Isocyanate (wt%)
  - Isocyanate (mmol NCO)
  - Isocyanate type
  - Ratio
  - Tin(II) octoate

  Lignin (wt%):
    VIF = 1061.28 [WARNING] HIGH multicollinearity - REMOVE

  Co-polyol (wt%):
    VIF = 1333.76 [WARNING] HIGH multicollinearity - REMOVE

  Co-polyol type (PTHF):
    VIF = 2.69 [OK] Low multicollinearity - KEEP

  Isocyanate (wt%):
    VIF = 882.01 [WARNING] HIGH multicollinearity - REMOVE

  Isocyanate (mmol NCO):
    VIF = 14.66 [WARNING] HIGH multicollinearity - REMOVE

  Isocyanate type:
    VIF = 27.22 [WARNING] HIGH multicollinearity - REMOVE

  Ratio:
    VIF = 29.80 [WARNING] HIGH multicollinearity - REMOVE

  Tin(II) octoate:
    VIF = 16.33 [WARNING] HIGH multicollinearity - REMOVE

[OK] VIF plot saved

================================================================================
FEATURE REDUCTION RECOMMENDATIONS
================================================================================

1. HIGH multicollinearity (VIF > 10): MUST REMOVE
   [X] Co-polyol (wt%) (VIF = 1333.76)
   [X] Lignin (wt%) (VIF = 1061.28)
   [X] Isocyanate (wt%) (VIF = 882.01)
   [X] Ratio (VIF = 29.80)
   [X] Isocyanate type (VIF = 27.22)
   [X] Tin(II) octoate (VIF = 16.33)
   [X] Isocyanate (mmol NCO) (VIF = 14.66)
✓ Step 1 complete

================================================================================
STEP 2/3: Running Fixed Stacking Ensemble...
Traceback (most recent call last):
  File "<string>", line 37, in <module>
    print('\u2713 Fixed Stacking complete')
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sacaru\AppData\Local\Python\pythoncore-3.14-64\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>
Running fixed stacking with 7 features...

================================================================================
STARTING RUN 1/1
================================================================================

Testing with 1000 estimators...

Running stacking with 1000 estimators using proper OOF predictions...
  Processing outer fold 1/10...
  Processing outer fold 2/10...
  Processing outer fold 3/10...
  Processing outer fold 4/10...
  Processing outer fold 5/10...
  Processing outer fold 6/10...
  Processing outer fold 7/10...
  Processing outer fold 8/10...
  Processing outer fold 9/10...
  Processing outer fold 10/10...
  Training final models on full dataset for deployment...
  [OK] New best MAE: 16.3840 deg C

[OK] Results saved to stacking_results_fixed_run_1.csv
Fixed models and scalers from run 1 saved successfully.
✓ Step 2 complete

================================================================================
STEP 3/3: Running Cascade Model...
Traceback (most recent call last):
  File "<string>", line 27, in <module>
    print('\u2713 Cascade Model complete')
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sacaru\AppData\Local\Python\pythoncore-3.14-64\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>
Running two-stage cascade analysis...

================================================================================
TWO-STAGE CASCADE MODEL - COMPLETE ANALYSIS
================================================================================

This addresses the swelling ratio circular dependency issue.
We compare three approaches:
  1. Baseline: Formulation only -> Tg
  2. Stage 1: Formulation -> Swelling
  3. Stage 2: Formulation + Predicted Swelling -> Tg

================================================================================

Dataset: 136 samples
Formulation features: ['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 'Isocyanate type', 'Tin(II) octoate']
Targets: Swelling ratio (%), Tg (deg C)


================================================================================
BASELINE: FORMULATION ONLY -> Tg (No Swelling)
================================================================================
Input features: ['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 'Isocyanate type', 'Tin(II) octoate']
Target: Tg (deg C)
Samples: 136

  Processing fold 1/10...
  Processing fold 2/10...
  Processing fold 3/10...
  Processing fold 4/10...
  Processing fold 5/10...
  Processing fold 6/10...
  Processing fold 7/10...
  Processing fold 8/10...
  Processing fold 9/10...
  Processing fold 10/10...

[OK] Baseline Complete:
  Validation MAE: 17.07 deg C
  Validation R2: 0.2858

================================================================================
STAGE 1: FORMULATION -> SWELLING RATIO
================================================================================
Input features: ['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 'Isocyanate type', 'Tin(II) octoate']
Target: Swelling Ratio (%)
Samples: 136

  Processing fold 1/10...
  Processing fold 2/10...
  Processing fold 3/10...
  Processing fold 4/10...
  Processing fold 5/10...
  Processing fold 6/10...
  Processing fold 7/10...
  Processing fold 8/10...
  Processing fold 9/10...
  Processing fold 10/10...

[OK] Stage 1 Complete:
  Validation MAE: 24.83%
  Validation R▒: 0.6685

================================================================================
STAGE 2: FORMULATION + PREDICTED SWELLING -> Tg
================================================================================
Input features: ['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 'Isocyanate type', 'Tin(II) octoate'] + Predicted Swelling
Target: Tg (deg C)
Samples: 136

  Processing fold 1/10...
  Processing fold 2/10...
  Processing fold 3/10...
  Processing fold 4/10...
  Processing fold 5/10...
  Processing fold 6/10...
  Processing fold 7/10...
  Processing fold 8/10...
  Processing fold 9/10...
  Processing fold 10/10...

[OK] Stage 2 Complete:
  Validation MAE: 16.67 deg C
  Validation R2: 0.2962

================================================================================
RESULTS COMPARISON
================================================================================
                                           Model  MAE Validation  MAE Train  R▒ Validation  R▒ Train  Generalizability
                      Baseline: Formulation Only       17.067140  16.934070       0.285811  0.340898          0.133071
                    Stage 1: Swelling Prediction       24.826937  23.100201       0.668527  0.741668          1.726736
Stage 2: Tg Prediction (with predicted swelling)       16.665615  16.557797       0.296208  0.372848          0.107817

================================================================================
INTERPRETATION
================================================================================

1. BASELINE (Formulation Only):
   - MAE: 17.07 deg C
   - This is the practical model (no synthesis required)
   - Lower accuracy but truly predictive

2. STAGE 1 (Swelling Prediction):
   - MAE: 24.83%
   - Predicts swelling from formulation
   - Enables cascade approach

3. STAGE 2 (Cascade Model):
   - MAE: 16.67 deg C
   - Uses predicted swelling (not actual)
   - Better than baseline, still fully predictive

4. CASCADE IMPROVEMENT:
   - Reduction in MAE: 0.40 deg C (2.4%)
   - Achieved without requiring synthesis first!

[OK] Results saved to 'cascade_model_results.csv'
[OK] Models saved to 'stage1_swelling_models.joblib' and 'stage2_tg_models.joblib'
✓ Step 3 complete

================================================================================
✓ ALL ANALYSES COMPLETE!
================================================================================

Generated files:
-rw-r--r-- 1 AIXTRON+sacaru 4096 2.2K Feb 15 16:43 Fixed_Stacking_Results.csv
-rw-r--r-- 1 AIXTRON+sacaru 4096  198 Feb 15 13:07 Reduced_Feature_Set.txt
-rw-r--r-- 1 AIXTRON+sacaru 4096  25K Feb 15 16:41 VIF_Analysis.pdf
-rw-r--r-- 1 AIXTRON+sacaru 4096 161K Feb 15 16:41 VIF_Analysis.png
-rw-r--r-- 1 AIXTRON+sacaru 4096  63K Feb 15 16:41 VIF_Analysis.svg
-rw-r--r-- 1 AIXTRON+sacaru 4096  295 Feb 15 13:07 VIF_Analysis_Results.csv
-rw-r--r-- 1 AIXTRON+sacaru 4096  470 Feb 15 16:46 cascade_model_results.csv

Check the log files for details:
  - step1_vif.log
  - step2_stacking.log
  - step3_cascade.log
