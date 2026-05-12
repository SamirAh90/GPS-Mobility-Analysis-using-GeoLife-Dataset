# GPS Mobility Mode Classification Using the GeoLife Dataset

## Assignment 2

**Authors:** Samir Ahmad, Roni Chaikho, Hangama Dehati, Ludwig Lindfors (Grupp 5)  
**Course:** Web-based Geographic Information Systems / Spatial Data and GIS  
**Date:** 2026-05-03  
**Professor/Examinator:** Vijay Paidi

---

## Abstract

This report presents a comprehensive machine learning pipeline for classifying transportation modes using GPS trajectory data from the Microsoft GeoLife dataset. The study aims to accurately identify four primary mobility modes—walking, cycling (bike), bus, and car—by engineering and evaluating seven spatio-temporal features. Three supervised classification tree-based ensemble models were developed and compared: Random Forest, Gradient Boosting, and XGBoost. All three models successfully exceeded the target accuracy threshold of 80%. The Random Forest classifier achieved the highest overall performance with an accuracy of **83.32%** and a weighted F1-score of **0.830**. The analysis demonstrates that speed and acceleration metrics are the most informative features for transport mode detection. However, distinguishing between buses and cars, as well as between bicycles and pedestrians, remains a complex challenge due to overlapping kinematic profiles in dense urban environments.

---

## 1. Introduction

Understanding how people move through urban environments is fundamental to smart city planning, sustainable transport policy, and human mobility research. The widespread adoption of GPS-enabled mobile devices has made it possible to continuously log high-resolution movement trajectories. These trajectories create rich, large-scale datasets that capture individual travel behaviour with unprecedented detail. However, raw GPS data consists merely of sequential latitude, longitude, and timestamp coordinates. The key challenge in spatial data analytics is transforming these raw coordinates into meaningful, semantically rich information, such as the mode of transportation used by the individual.

The **GeoLife dataset** (Zheng et al., 2008–2012), published by Microsoft Research Asia, serves as the foundation for this study. It contains extensive GPS trajectories collected from 182 users over a period of five years, primarily centered in Beijing, China. A subset of these users diligently annotated their trajectories with specific transportation mode labels, making the dataset highly suitable for supervised machine learning classification tasks. The four primary transport modes evaluated in this study are: **walk**, **bus**, **bike**, and **car**.

The objective of this project is to develop and evaluate machine learning models capable of automatically inferring these four transportation modes from engineered GPS features. Velocity, acceleration, turning behaviour, and stop patterns differ fundamentally between modes. For example, a pedestrian moves slowly with frequent direction changes, a cyclist moves moderately fast with smooth turns, a bus moves along fixed routes with regular stops, while a car moves at high speeds with relatively smooth acceleration profiles.

To contextualise the data, _Figure 1 (plots/class_distribution.png)_ illustrates the background distribution of the dataset. The dataset is moderately imbalanced, heavily dominated by walking segments (47.8%), followed by bus (20.3%), bike (16.4%), and car (15.5%). This imbalance reflects the natural frequency of short pedestrian trips in urban daily life but requires careful consideration during model evaluation.

This report is structured in the IMRAD format and addresses the following core questions:

1. Which supervised classification model performs better, and why?
2. Which transportation modes are difficult to classify?
3. What features contribute most to the classification?
4. What are the inherent limitations of each applied model?

---

## 2. Method

### 2.1 Dataset and Preprocessing

The pre-processed dataset used in this study consists of **11,326 trajectory segments**. Each segment represents a continuous period of movement labelled with one of the four target transportation modes.

To ensure robust model evaluation, the dataset was divided into training and testing sets using a **stratified 80/20 random split**.

- **Training set:** 9,060 segments (80%)
- **Test set:** 2,266 segments (20%)

The use of stratification is critical; it guarantees that the original class distribution (e.g., 47.8% walking, 15.5% car) is perfectly preserved in both the training and testing sets. This prevents minority classes (like cars and bikes) from being underrepresented during the evaluation phase, which could otherwise lead to misleading accuracy metrics. Furthermore, the features were standardised using `StandardScaler` (zero mean, unit variance). While tree-based ensemble models are generally scale-invariant, standardisation ensures uniformity and consistency in the pipeline.

### 2.2 Feature Engineering

Raw GPS points cannot be fed directly into standard tabular classifiers. Therefore, each trajectory segment was characterised by **7 engineered spatio-temporal features**, selected based on established literature (Li et al., 2021) and the physical characteristics of movement:

1. **`avg_speed_mps`**: The mean speed across the segment (meters/second).
2. **`max_speed_mps`**: The maximum instantaneous speed recorded in the segment (high-speed behaviour).
3. **`distance_m`**: The total distance travelled during the segment (meters).
4. **`std_acc_mps2`**: The standard deviation of acceleration, representing variance in kinetic energy.
5. **`avg_turn_deg`**: The mean turning angle (degrees) between consecutive GPS points.
6. **`stop_ratio`**: The fraction of time the segment is completely stationary.
7. **`std_speed_mps`**: The standard deviation of speed, capturing the stop-and-go nature of traffic.

These features operationalise physical intuitions. For example, `stop_ratio` is designed to capture the frequent stops inherent to bus routes, while `max_speed_mps` naturally separates motorised vehicles on highways from pedestrians.

### 2.3 Feature Importance Analysis

Before training complex models, an exploratory analysis of feature importance was conducted to validate the engineered variables. _Figure 2 (plots/feature_boxplots.png)_ illustrates the discriminative power of the features through class-wise distribution analysis.

The boxplots reveal that `avg_speed_mps` and `max_speed_mps` possess immense discriminative power; for instance, cars and buses consistently record maximum speeds that far exceed human physical limitations (walking and cycling). Meanwhile, `std_acc_mps2` highlights the structural differences in kinetic energy—cars exhibit sharp acceleration variance in traffic, whereas pedestrians maintain smooth, low-variance acceleration. `distance_m` effectively isolates long vehicular trips from short walks. This exploratory step visually confirmed that the 7 selected features carry a strong, separable predictive signal, justifying their use in the subsequent machine learning models.

### 2.4 Classification Models

Three advanced supervised classification algorithms were trained to evaluate their effectiveness:

1. **Random Forest (RF):** An ensemble learning method that constructs a multitude of decision trees during training. Each tree is trained on a bootstrapped sample of the data and a random subset of features. The final prediction is determined by majority voting. It is highly robust to overfitting and handles non-linear relationships natively.
2. **Gradient Boosting (GB):** Unlike the independent trees in Random Forest, Gradient Boosting builds trees _sequentially_. Each new tree is specifically designed to correct the residual errors made by the previous ensemble, guided by the negative gradient of the loss function.
3. **XGBoost (Extreme Gradient Boosting):** An advanced, highly optimised implementation of gradient boosting. It uses second-order derivatives to guide the boosting process and incorporates L1/L2 regularisation to penalise complex models, making it one of the most powerful algorithms for tabular data.

### 2.5 Evaluation Metrics

Model performance was evaluated on the held-out test set using Accuracy, Precision, Recall, and the weighted F1-score. Additionally, confusion matrices and Receiver Operating Characteristic (ROC) curves were generated to provide deeper insight into class-specific performance and misclassification patterns.

---

## 3. Results

### 3.1 Overall Model Performance: Which model performs better?

The overall performance of the three classifiers is summarised in _Figure 3 (plots/metrics_comparison.png)_. The results on the unseen test set (2,266 segments) are as follows:

| Model             | Accuracy   | Weighted F1-Score |
| ----------------- | ---------- | ----------------- |
| **Random Forest** | **83.32%** | **0.830**         |
| Gradient Boosting | 82.97%     | 0.827             |
| XGBoost           | 82.52%     | 0.822             |

All three models successfully exceeded the project's threshold of 80% accuracy. **Random Forest performed the best**, marginally outperforming both Gradient Boosting and XGBoost.

**Why did Random Forest perform better?**
While Gradient Boosting and XGBoost are theoretically more powerful due to their sequential error-correction mechanisms, they are also highly sensitive to hyperparameter tuning and prone to overfitting on datasets with inherent noise. GPS data collected in urban canyons (like Beijing) is notoriously noisy due to multipath errors and signal drops. Random Forest's mechanism of independent bagging (training deep trees on random subsets of data and features) makes it exceptionally robust to this kind of noisy data. It successfully captured the non-linear boundaries between modes without overfitting to the specific noise patterns in the training set, whereas the boosting models slightly over-optimised on the training residuals, leading to a minor drop in generalisation accuracy on the test set.

### 3.2 Difficulty in Classifying Modes: Which modes are difficult to classify?

To understand class-specific difficulties, we must analyse the Per-Class F1-scores (_Figure 4, plots/per_class_f1.png_) and the Confusion Matrices (_Figures 5-7: plots/confusion_matrix_rf.png, plots/confusion_matrix_gb.png, and plots/confusion_matrix_xgb.png_). The results for the best-performing Random Forest model are:

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Walk  | 0.89      | 0.78   | 0.83     |
| Bus   | 0.72      | 0.68   | 0.70     |
| Bike  | 0.81      | 0.70   | 0.75     |
| Car   | 0.87      | 0.96   | 0.91     |

**The most difficult modes to classify are Bus (F1 = 0.70) and Bike (F1 = 0.75).**

The confusion matrix analysis reveals distinct patterns that explain these difficulties:

1. **Bus vs. Car:** The most common misclassification occurs between buses and cars. A significant number of true bus segments are incorrectly predicted as cars. This occurs because, on open roads or highways, a bus's speed and acceleration profile is virtually identical to that of a car. If a bus segment does not happen to include a stop at a bus station, the model lacks the necessary kinetic evidence to differentiate it from a car.
2. **Bike vs. Walk:** Slower cycling segments are frequently confused with faster walking segments. In congested urban areas, a cyclist moving slowly through traffic or pushing their bike across an intersection generates a speed and turning profile that heavily overlaps with a pedestrian.

Conversely, **Car** is the easiest to classify (F1 = 0.91). Cars routinely achieve high maximum speeds (`max_speed_mps`) and cover long distances (`distance_m`) that are physically impossible for pedestrians or cyclists, making them easily distinguishable. _Figure 8 (plots/roc_curves.png)_ confirms this, showing a near-perfect Area Under the Curve (AUC = 0.982) for cars.

---

## 4. Discussion

### 4.1 What features contribute most to classification?

The classification success is heavily dependent on the engineered features. The results demonstrate that **speed and acceleration metrics** contribute the most to the classification logic.

Variables such as `max_speed_mps` and `avg_speed_mps` act as the primary physical discriminants. Human physical limitations dictate that pedestrians rarely exceed 2 m/s, and cyclists generally travel between 3–6 m/s. Therefore, any segment featuring speeds consistently above 10 m/s is immediately identifiable as a motorised vehicle (car or bus).

Furthermore, **`std_acc_mps2` (acceleration variance)** and **`stop_ratio`** provide secondary, highly vital discriminative power. A car driving through city traffic exhibits sharp peaks in acceleration and deceleration, leading to a high `std_acc_mps2`. Meanwhile, `stop_ratio` is crucial for identifying buses, which are characterised by periodic, prolonged stationary periods at designated bus stops.

Finally, spatial features like **`distance_m`** capture trip context—cars and buses typically undertake longer journeys than walkers—while **`avg_turn_deg`** helps the model differentiate between the highly erratic, grid-independent movement of a wandering pedestrian and a car constrained to straight road networks.

### 4.2 What are the limitations of each model?

Despite the high accuracy achieved, each machine learning model presents specific limitations:

**Limitations of Random Forest:**

- **Inability to Extrapolate:** Random Forest can only predict values within the range of its training data. If deployed in a new city where highway speeds are significantly higher than in Beijing, the model cannot extrapolate beyond the maximum speeds it observed in the training set.
- **Memory Consumption:** An ensemble of 300 deep decision trees requires a significant amount of memory. While manageable on this dataset, it becomes computationally expensive to deploy on edge devices (like mobile phones) for real-time inference.

**Limitations of Gradient Boosting:**

- **Training Time and Sequential Bottleneck:** Because trees must be built sequentially, Gradient Boosting cannot be easily parallelised across CPU cores during training. It took significantly longer to train than the Random Forest.
- **Sensitivity to Noise:** By constantly focusing on the hardest-to-predict samples, standard Gradient Boosting is highly susceptible to fitting to GPS noise (e.g., satellite jumps), which likely caused its slight underperformance on the test set.

**Limitations of XGBoost:**

- **Hyperparameter Sensitivity:** XGBoost has a vast array of hyperparameter configurations (learning rate, depth, lambda, alpha, subsample). Achieving optimal performance requires exhaustive grid search cross-validation. Without perfect tuning, it is prone to severe overfitting on datasets of this size.
- **Complexity:** The mathematical complexity of XGBoost reduces interpretability compared to examining a single decision tree.

**General Dataset Limitations:**
The most prominent limitation across all models is the dataset itself. The GeoLife data is geographically restricted to Beijing, China. Urban infrastructure, traffic congestion, and travel behaviours vary wildly globally. A model trained on Beijing's traffic patterns may struggle to classify transport modes in rural areas or cities with different public transit dynamics. Furthermore, treating trajectories as independent, tabular segments discards valuable sequential and temporal context that could have helped resolve the ambiguity between buses and cars.

---

## 5. Conclusion

This project successfully developed an end-to-end spatial data analytics pipeline capable of classifying transportation modes from raw GPS trajectories. By extracting seven physically meaningful spatio-temporal features, all tested models (Random Forest, Gradient Boosting, XGBoost) achieved robust accuracies exceeding 82%. Random Forest proved to be the superior model (83.32% accuracy), demonstrating resilience against the inherent noise found in urban GPS recordings. While speed and acceleration features effectively isolate motorised from non-motorised transport, distinguishing between buses and cars remains a persistent challenge due to shared road networks and overlapping kinematic profiles. Future work could address these limitations by integrating geographic context (such as proximity to mapped bus routes via GIS integration) or by employing deep sequential models (like LSTMs) to capture the full temporal dynamics of a user's journey.

---

## References

1. Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008). Understanding mobility based on GPS traces. _Proceedings of the 10th international conference on Ubiquitous computing_, 312-321.
2. Zheng, Y., Liu, L., Wang, L., & Xie, X. (2008). Learning transportation mode from raw GPS data for geographic applications on the web. _Proceedings of the 17th international conference on World Wide Web_, 247-256.
3. Li, J., Pei, X., Wang, X., Yao, D., Zhang, Y., & Yue, Y. (2021). Transportation Mode Identification with GPS Trajectory Data and GIS Information. _Tsinghua Science and Technology_, 26(4), 403-416.
4. Breiman, L. (2001). Random forests. _Machine learning_, 45(1), 5-32.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining_, 785-794.
6. Microsoft Research Asia. GeoLife GPS Trajectories Dataset. User Guide and Documentation. https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
