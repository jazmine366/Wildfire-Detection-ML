# 🔥 Wildfire Detection (Tabular DataSet)

This project builds and compares machine learning models to classify **fire vs no-fire** using a small satellite based dataset. The goal is to analyze how different models behave under **severe class imbalance** and noisy real world tabular features.

---
## 1. Dataset & Preprocessing

**File:** `WildFire_DataSet.csv`  
**Features:**
- **NDVI** – vegetation index (-1 to 1)
- **LST** – land surface temperature (scaled Kelvin)
- **BURNED_AREA** – burned land in hectares  
- **CLASS** – fire (1) / no fire (0)

**Preprocessing Steps**
- Loaded CSV, checked for missing values  
- Encoded CLASS (fire/no_fire → 1/0)  
- Normalized features using **MinMaxScaler**  
- Train/Test split: 80/20 (`random_state=42`)

**Class distribution:**  
- Fire: **275**  
- No Fire: **68**  
  --> Strong **class imbalance (4:1)**

---
## 2. Models Trained & Results

### **Model 1 — Logistic Regression**
- Accuracy: **~80%**
- Recall (Fire): **~99%**
- Recall (No Fire): **~6%**

**Summary:**  
Predicts “fire” almost every time → accuracy looks good but recall for class 0 is extremely poor. Meaning model doesn't really learn the pattern rather it just outputs fire.

---

### **Model 2 — Random Forest BEST MODEL (class_weight="balanced")**
- Accuracy: **~86%**
- Recall (Fire): **~94%**
- Recall (No Fire): **~51%**

**Summary:**  
Best performing model. Handles imbalance well and significantly improves no fire detection. 
Recall > Accuracy in importance, visualize by classification report, confusion matrixs

---

### **Model 3 — Neural Network (MLP)**
Architecture: `32 → 16 → 1` (ReLU, ReLU, Sigmoid)

- Accuracy: **~80%**
- Recall (Fire): **~96%**
- Recall (No Fire): **~15%**

**Summary:**  
Still biased toward predicting fire. Accuracy unstable between ~60% to 80% before setting deterministic seeds.

---

## 3. Key Finding: Imbalance Dominates Model Behavior

- Fire cases heavily outweigh no-fire cases  
- **Recall** is the important metric rather then accuracy 
- Logistic Regression & MLP heavily favor the majority class as they have unbalanced variable weight
- Random Forest is the only model with balanced recall

**Conclusion:**  
Model performance is limited by the **dataset**, not algorithm choice.

---

## 4. Conclusions & Future Improvements

- Dataset imbalance leads to biased predictions  
- High accuracy does not mean good performance  
- Random Forest outperforms other methods under imbalance  
- Collecting more **no-fire** data would dramatically improve results  
- Additional environmental features are needed for a realistic wildfire classifier (humidity, wind, vegetation moisture)
