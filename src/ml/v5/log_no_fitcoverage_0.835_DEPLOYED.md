PlantLeaf v5 — SVM Training
  CSV            :  /Users/tommy/PlantLeaf_dev/Analisi/v5/SVM_Training/Dataset_20June2026.csv
  Kernels        :  ['rbf']
  Recall target  :  0.835
  Set B sessions :  ['Aloe_acqua50ml_misurazione1_11032026_09']
  Noise filter   :  R²>0.1, SPR<100.0
  Excluded       :  ['fit_coverage']  (16/17 features active)
  Seed           :  42

Labeled rows loaded:  285
  label=1 (clicks) :  91
  label=0 (noise)  :  194
  Sessions         :  38

Noise pre-filter  R²>0.1, SPR<100.0:
  noise samples:  194 → 192
  class ratio clicks:noise = 1:2.1

Set B (held-out test): 26 rows  (clicks=16, noise=10)
  Sessions: ['Aloe_acqua50ml_misurazione1_11032026_09']
Set A (training):      257 rows  (clicks=75, noise=182)
  Sessions: ['Calancola_insetti_coccinelle_misurazione1_09032026_10.52mattina', 'Carnivora_insetti_misurazione1_13032026_9.44', 'aloe_6jan', 'aloe_afternowater_21feb', 'aloe_meccanico2buchi_pomeriggio1Mar', 'aloe_solopianta_1_27feb_pomeriggiotardi', 'aloe_solopianta_2_27feb_sera', 'aloe_stimolomeccanico_misurazine1_cameraricy_270420261210', 'ambienteesterno-balcon-solorumori-misurazione3-150420260903', 'ambienteesterno-balcone-rumoristrada-uccelli-lavori-14.04.2026 10.00', 'balcone_06042026_1128', 'balcone_26042026_1512_primogommapiuma', 'balcone_misurazione2_06042026_1242', 'cactus_acqua_misurazione1_13032026_08.13_final', 'cactus_stimolomeccanico_misurazione4_16032026_12.04', 'cactus_stimolomeccanico_misurazione5_16032026_13.30', 'calancola_acqua200ml_misurazione1_09032026_9.40', 'solopianta_aloe_misurazione4_11032026_08.03_final', 'solopianta_aloe_misurazione5_12032026_8.07mattina_final', 'solopianta_cactus_misurazione3_04032026_12.30mattino_final', 'solopianta_cactus_misurazione4_12032026_9.11_final', 'solorumore_cameraricy_misurazione1_280420261012', 'solorumore_cameraricy_misurazione2_290420261024', 'solorumore_cameraricy_misurazione3_30042026915', 'solorumore_cameraricy_misurazione4_040520260848', 'solorumore_cameraricy_misurazione5_070520260849', 'solorumore_misurazione6_cameraricy_120520260853', 'stanzavuota_nessunostimolo_misurazione1_02032026_08.15mattina_final', 'stanzavuota_nessunostimolo_misurazione2_03032026_8.17mattina_final', 'stanzavuota_nessunostimolo_misurazione4_05032026_14.40_final', 'stimolomeccanico_aloe_misurazione1_03032026_13.49', 'stimolomeccanico_aloe_misurazione2_10032026_15.00', 'stimolomeccanico_aloe_misurazione3_11032026_10.46', 'stimolomeccanico_cactus_misurazione1_03032026_10.39mattina_final', 'stimolomeccanico_cactus_misurazione2_05032026_13.12mattina_final', 'stimolomeccanico_cactus_misurazione3_11032026_10.00', 'test_aloe_1']

============================================================
  Kernel: RBF
============================================================
  Features used (16): ['peak_SNR', 'pre_SNR', 'post_SNR', 'rise_time_ms', 'fall_time_ms', 'asymmetry_integral', 'ZCR_pre', 'ZCR_click', 'ZCR_post', 'kurtosis', 'centroid_shift_hz', 'tau_ms', 'R2', 'SPR', 'R_spectral', 'FPE_hz']
  GridSearchCV: 20 combinations × 5 folds = 100 fits
  Scoring: recall (primary metric)  |  Groups: session_id

  Best params    :  {'C': 50, 'gamma': 0.01}
  Best CV recall :  0.728
  Computing out-of-fold probabilities (5 more fits)...

  Cross-validated metrics  (threshold = 0.50):
  Confusion matrix  TP=35  FP=18  FN=40  TN=164
  Recall       [PRIMARY] :  0.467   (35/75 clicks detected)
  Precision              :  0.660
  Specificity            :  0.901
  F1                     :  0.547
  AUC-ROC                :  0.835
  Accuracy               :  0.774   (not primary metric)

  Cross-validated metrics  (threshold = 0.264, target recall ≥ 0.835):
  Confusion matrix  TP=63  FP=47  FN=12  TN=135
  Recall       [PRIMARY] :  0.840   (63/75 clicks detected)
  Precision              :  0.573
  Specificity            :  0.742
  F1                     :  0.681
  AUC-ROC                :  0.835
  Accuracy               :  0.770   (not primary metric)

  Feature importance (rbf):
    Computing permutation importance on Set A (n_repeats=15)...
     1. fall_time_ms            Δrecall=+0.119 ± 0.028  ███████████████████
     2. peak_SNR                Δrecall=+0.104 ± 0.021  █████████████████
     3. FPE_hz                  Δrecall=+0.065 ± 0.032  ██████████
     4. post_SNR                Δrecall=+0.064 ± 0.028  ██████████
     5. pre_SNR                 Δrecall=+0.061 ± 0.027  ██████████
     6. tau_ms                  Δrecall=+0.056 ± 0.020  █████████
     7. rise_time_ms            Δrecall=+0.047 ± 0.020  ███████
     8. kurtosis                Δrecall=+0.044 ± 0.034  ███████
     9. ZCR_post                Δrecall=+0.032 ± 0.023  █████
    10. asymmetry_integral      Δrecall=+0.025 ± 0.020  ████
    11. ZCR_pre                 Δrecall=+0.023 ± 0.022  ███
    12. ZCR_click               Δrecall=+0.017 ± 0.013  ██
    13. R_spectral              Δrecall=+0.015 ± 0.018  ██
    14. R2                      Δrecall=+0.010 ± 0.021  █
    15. SPR                     Δrecall=+0.006 ± 0.019  █
    16. centroid_shift_hz       Δrecall=-0.002 ± 0.016  
    (Δrecall: drop in recall when feature is shuffled; larger = more important)
    Note: re-run permutation_importance() on Set B for the published result.

============================================================
  Set B — held-out test set evaluation
============================================================
  Threshold used:  0.264
  Confusion matrix  TP=13  FP=1  FN=3  TN=9
  Recall       [PRIMARY] :  0.812   (13/16 clicks detected)
  Precision              :  0.929
  Specificity            :  0.900
  F1                     :  0.867
  AUC-ROC                :  0.925
  Accuracy               :  0.846   (not primary metric)

  Per-session breakdown (Set B):
    Session                              clicks  detected  FP    recall
    ----------------------------------------------------------------------
    Aloe_acqua50ml_misurazione1_11032026_09      16        13     1  0.81

============================================================
  Summary
============================================================
  Kernel      CV AUC-ROC  Threshold
  rbf         0.835       0.264

  Best kernel: rbf  (CV AUC-ROC = 0.835)

  Model saved:  /Users/tommy/PlantLeaf_dev/PlantLeaf-Desktop-App/docs/autoclick/v5/pkl/plantleaf_svm_v5_allfeatures_0.835.pkl

  Inference usage:
    import joblib, numpy as np
    model = joblib.load('/Users/tommy/PlantLeaf_dev/PlantLeaf-Desktop-App/docs/autoclick/v5/pkl/plantleaf_svm_v5_allfeatures_0.835.pkl')
    pipe  = model['pipeline']
    thr   = model['threshold']   # 0.264
    # X: (n, 17) array, columns = model['features']
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= thr).astype(int)  # 1=click, 0=noise