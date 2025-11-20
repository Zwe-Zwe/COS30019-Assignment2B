## Overall Modelling Progress

This project explores several different models for classifying incident severity into three classes (`01-minor`, `02-moderate`, `03-severe`) using the same dataset under `data3a/` and a shared training/validation split. The models are:

- A **baseline CNN** trained from scratch.
- A **HOG + SVM** classical vision pipeline.
- A **ResNet18 feature extractor + Gradient Boosted Trees (GBT)** classifier.
- A **transfer-learning ResNet18 CNN** fine-tuned on the dataset.

All models are evaluated on the same validation set of 248 images, and their behaviour is tracked using logs in `training_logs/` and checkpoints in `models/`.

---

## 1. Baseline CNN (`train_baseline_cnn.py`)

### Model description

The baseline model is a **simple convolutional neural network (CNN)** implemented in `train_baseline_cnn.py` as `SimpleCNN`. It is a lightweight architecture designed to establish a reference point for more advanced models.

Key characteristics:
- **Input:** 3-channel colour images.
- **Feature extractor:** A stack of convolutional blocks, each consisting of:
  - 2D convolution (`Conv2d`)
  - Batch Normalisation (`BatchNorm2d`)
  - ReLU activation
  - Max-pooling (`MaxPool2d`) to reduce spatial size
- **Classifier head:** Fully connected layers producing logits for the three severity classes.

### Training behaviour and results

Using the later run `training_logs/baseline_cnn_20251120-004353.log` (20-epoch run with early stopping):

- Training loss decreases steadily from **3.63** to around **0.89**.
- Validation loss improves from **1.05** to around **0.84–0.86** with some fluctuations.
- Validation accuracy improves from **46.37%** to a **best of 62.50%**.
- Early stopping triggers at epoch 16 after no further improvement.

Final best-checkpoint metrics:

- **Best validation loss:** 0.8578
- **Best validation accuracy:** 62.50%

Per-class performance (validation):

- `01-minor`: precision **0.64**, recall **0.85**, F1 **0.73**
- `02-moderate`: precision **0.52**, recall **0.21**, F1 **0.30**
- `03-severe`: precision **0.64**, recall **0.76**, F1 **0.70**

Overall:

- **Accuracy:** 0.62
- **Macro F1:** 0.58
- **Weighted F1:** 0.59

**Interpretation:**

- The baseline CNN learns useful features and performs well on `01-minor` and `03-severe`, especially with high recall for severe cases.
- It struggles with `02-moderate` (particularly recall), indicating that moderate incidents are frequently confused with minor or severe.
- This model provides the **deep-learning baseline** for later comparisons.

---

## 2. HOG + SVM (`train_hog_svm.py`)

### Model description

This model uses a **classical computer vision pipeline**:

- Images are resized, converted to grayscale, and transformed into **Histogram of Oriented Gradients (HOG)** feature vectors using `skimage.feature.hog`.
- A **StandardScaler + SVM** classifier with RBF kernel is trained on these HOG features using scikit-learn.
- The script `train_hog_svm.py` logs results and saves the trained model to `models/hog_svm.pkl`.

### Training behaviour and results

From `training_logs/hog_svm_20251120-004541.log`:

- HOG features are computed for **1383 training** images and **248 validation** images, giving feature vectors of dimension 17,328.
- A single SVM fit is performed (no epoch-based training like in neural networks).

Final validation performance:

- **Validation accuracy:** 59.68%

Per-class metrics:

- `01-minor`: precision **0.64**, recall **0.60**, F1 **0.62**
- `02-moderate`: precision **0.45**, recall **0.49**, F1 **0.47**
- `03-severe`: precision **0.70**, recall **0.68**, F1 **0.69**

Overall:

- **Accuracy:** 0.60
- **Macro F1:** 0.59
- **Weighted F1:** 0.60

**Interpretation:**

- HOG+SVM achieves **similar overall accuracy** to the baseline CNN but with a **more balanced performance** across all three classes.
- In particular, the `02-moderate` class is handled **better** than by the baseline CNN (higher recall and F1), suggesting that simple gradient-based features capture some mid-level severity cues that the shallow CNN misses.
- This model is strong for a classical approach and forms a useful non-deep-learning baseline.

---

## 3. ResNet18 Features + Gradient Boosted Trees (`train_resnet_features_gbt.py`)

### Model description

This approach combines **deep features** from a pretrained ResNet18 with a **gradient-boosted tree classifier**:

- A pretrained **ResNet18** (ImageNet weights) from `torchvision.models` is used as a **fixed feature extractor**. The final classification layer is removed, so we take the pooled 512-dimensional feature vector for each image.
- These feature vectors for all images are extracted using `extract_features` and saved as NumPy arrays.
- A **HistGradientBoostingClassifier** (scikit-learn) is trained on the ResNet features.
- The script `train_resnet_features_gbt.py` logs metrics and writes a model checkpoint `models/resnet_gbt.pkl` that stores the trained trees and metadata.

### Training behaviour and results

From `training_logs/resnet_gbt_20251120-004528.log`:

- ResNet18 features are extracted for **1383 training** and **248 validation** images, each mapped to a 512-dimensional feature vector.
- The HistGradientBoostingClassifier is trained with the specified hyperparameters (e.g. `max_depth=3`, `learning_rate=0.05`, `max_iter=200`).

Final validation performance:

- **Validation accuracy:** 68.55%

Per-class metrics:

- `01-minor`: precision **0.74**, recall **0.78**, F1 **0.76**
- `02-moderate`: precision **0.51**, recall **0.48**, F1 **0.49**
- `03-severe`: precision **0.78**, recall **0.77**, F1 **0.77**

Overall:

- **Accuracy:** 0.69 (68.55%)
- **Macro F1:** 0.67
- **Weighted F1:** 0.68

**Interpretation:**

- This hybrid model substantially **outperforms both the baseline CNN and the HOG+SVM** in terms of overall accuracy and F1.
- It offers **good performance on all three classes**, including a clear improvement on `02-moderate` compared to the baseline CNN.
- The results show that leveraging **pretrained deep features + powerful tree-based classifiers** is a very effective strategy on this dataset.

---

## 4. Transfer-Learning ResNet18 CNN (`train_transfer_cnn.py`)

### Model description

This model performs **end-to-end transfer learning** using a pretrained ResNet18:

- `build_model` loads **ResNet18 with ImageNet weights**.
- By default, the backbone is **frozen** and only the final fully connected head is trained; with `--unfreeze`, the entire network can be fine-tuned.
- The final classification layer is replaced with a small head: `Dropout(0.4)` followed by a `Linear` layer to the 3 classes.
- A range of options exist for **data augmentation**, learning rate schedule (cosine or step), and early stopping, making this the most flexible deep model in the project.

### Training behaviour and results

From `training_logs/transfer_resnet18_20251120-004432.log`:

- Training runs for **12 epochs**.
- Training loss decreases from **1.07** to about **0.54**.
- Validation loss and accuracy fluctuate as the model trains, but there is a clear upward trend in accuracy.

Key checkpoints:

- Epoch 1: val_acc **62.10%**
- Epoch 5: val_acc **68.15%**
- Epoch 8: val_acc **68.95%**
- Epoch 11: val_acc **70.16%** (best)

Final best-checkpoint metrics:

- **Best validation loss:** 0.6663
- **Best validation accuracy:** 70.16%

Per-class performance (validation):

- `01-minor`: precision **0.70**, recall **0.84**, F1 **0.77**
- `02-moderate`: precision **0.58**, recall **0.40**, F1 **0.47**
- `03-severe`: precision **0.77**, recall **0.82**, F1 **0.79**

Overall:

- **Accuracy:** 0.70
- **Macro F1:** 0.68
- **Weighted F1:** 0.69

**Interpretation:**

- The transfer-learning ResNet18 is **the best-performing end-to-end CNN** in this project.
- It achieves slightly higher accuracy than the ResNet+GBT hybrid, with particularly strong recall for both `01-minor` and `03-severe`.
- Performance on `02-moderate` is still weaker than on the other two classes, but better than the baseline CNN.

---

## 5. Comparison of All Models

Summary of best validation results across models (on the same validation set):

| Model                              | Val Acc | Macro F1 | Notes |
|------------------------------------|--------:|---------:|-------|
| Baseline CNN (SimpleCNN)          | 62.50%  | 0.58     | Strong recall for severe; weak on moderate |
| HOG + SVM                         | 59.68%  | 0.59     | Balanced classical method; better moderate class than baseline CNN |
| ResNet18 features + GBT           | 68.55%  | 0.67     | Big jump from deep features + trees |
| Transfer-learning ResNet18 (CNN)  | 70.16%  | 0.68     | Best overall; high recall for minor and severe |

Key observations:

- Moving from **scratch CNN** → **HOG+SVM** → **ResNet features + GBT** → **transfer ResNet18** gives a **clear, consistent improvement** in overall accuracy and F1.
- All strong models (ResNet+GBT and transfer ResNet18) still find `02-moderate` the most challenging class, but they **substantially improve** on it compared to the baseline CNN.
- Pretrained representations from ResNet18 (either with trees or end-to-end fine-tuning) are **much more effective** than training a small CNN from scratch on this dataset.

---

## 6. Overall Conclusion

- The experiments demonstrate that **model choice has a large impact** on performance for incident severity classification.
- The **baseline CNN** is a useful starting point but is clearly outperformed by models that leverage **pretrained deep features**.
- The **HOG+SVM** approach provides a competitive classical baseline and handles the moderate class better than the baseline CNN.
- The **ResNet18 features + GBT** and **transfer-learning ResNet18** models deliver the **best overall results**, with the transfer-learning CNN achieving the highest validation accuracy (~70%).
- Future work should focus on:
  - Further improving performance on the **moderate** class (e.g. class weighting, focal loss, targeted augmentation).
  - Exploring deeper fine-tuning of the ResNet backbone and hyperparameter optimisation.
  - Evaluating robustness and generalisation on additional test data or different incident distributions.
