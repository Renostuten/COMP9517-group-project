## Project Status
### Last updated: November 4, 2025 (by Jiwoo)

**Completed**

- YOLO Data Loader (`load_yolo_dataset`) [cell 4]
- HOG Feature Extractor (`HOGFeatureExtractor`) [cell 5]
- SVM Classifier (`SVMClassifier`) [cell 6]


**TODO**

The current implementation only includes the Classifier. To meet the project requirements, we need to complete the Detector module and build the evaluation pipeline.

- Implement the detection module
    - Replace the current “ground-truth crop” approach with a sliding window logic suitable for real detection.
    - Once completed, compute mAP to evaluate detection performance as required by the spec.

- Implement `SVMClassifier.evaluate`
    - The `evaluate` method is currently empty (`pass`) [cell 6].

- Hyperparameter tuning
    - Find a good combination of HOG params [cell 5] and SVM params [cell 6].

- (Advanced : Evaluate robustness or test data unbalanced proble)

**Before start**
- Make sure everyone is using exactly the same package versions as specified in the 'environment.yml' file.
- Never ever push to the main branch.
- The **ml-pipeline** branch contains the initial pipeline implementation using HOG + SVM. For personal work, either create a separate branch based on this branch or, if you find Git difficult to use, simply share your work with Jiwoo (+ML team members) directly.

