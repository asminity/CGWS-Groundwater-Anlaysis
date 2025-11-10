# 💧 CGWS Groundwater Quality Analysis

### 🌍 Project Overview
This repository presents a **comprehensive groundwater quality analysis** performed using data from multiple Indian states.  
The project aims to **assess, visualize, and predict** groundwater safety using data science and machine learning techniques.  

It leverages **Python, pandas, seaborn, folium, and scikit-learn** to clean, analyze, and classify samples as *Safe* or *Unsafe* based on **BIS/WHO water quality standards**.

---

## 🧾 Objectives
- Evaluate groundwater quality across regions using chemical parameters.  
- Detect unsafe samples exceeding standard limits.  
- Identify key factors influencing contamination.  
- Develop a machine learning model to predict water safety.  
- Create interactive and visual outputs for interpretation and policy use.

---

## 📂 Repository Structure

| File / Folder | Description |
|----------------|-------------|
| **Groundwater.xlsx** | Source dataset containing groundwater parameters. |
| **report.ipynb** | Main Jupyter Notebook with full workflow (data cleaning, EDA, modeling, visualization). |
| **README.md** | Project documentation (this file). |
| **requirements.txt** | Dependencies and package versions required to run the notebook. |
| **map.html** | Interactive groundwater map generated with Folium. |
| **ph_distribution.png** | pH value distribution plot. |
| **unsafe_states.png** | Bar plot showing top unsafe states. |
| **correlation_heatmap.png** | Correlation matrix of groundwater parameters. |
| **confusion_matrix.png** | Confusion matrix of classification results. |
| **roc_curve.png** | ROC curve visualizing model performance. |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CGWS-Groundwater-Analysis.git
cd CGWS-Groundwater-Analysis
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the Notebook
bash
Copy code
jupyter notebook report.ipynb
5. View Results
Open map.html in your browser to explore safe/unsafe locations.

Generated .png plots will appear in your working directory.
```

## 📊 Project Workflow

### **1. Data Preprocessing**
- Missing values handled using **KNN Imputation**.  
- Removed columns with >30% missing data (e.g., **As, Fe, PO₄, U**).  
- Renamed and standardized column names and units.

---

### **2. Exploratory Data Analysis (EDA)**
- Visualized **pH** and **EC (Electrical Conductivity)** distributions using density plots.  
- Detected unsafe samples based on **BIS/WHO safe limits**.  
- Identified top unsafe states using frequency counts.

---

### **3. Correlation Analysis**
- Generated a **heatmap** to visualize relationships between key ions.  
- Found strong correlations among **EC, Na, Cl, and Hardness**, representing **salinity and mineralization processes**.

---

### **4. Geospatial Mapping**
- Created an **interactive map** using **Folium** with:
  - 🟢 **Green markers:** Safe samples  
  - 🔴 **Red markers:** Unsafe samples  
- Unsafe clusters are concentrated in **western, northern, and southern India**.

---

### **5. Machine Learning Model**
- Trained a **Random Forest Classifier** to predict water safety.  
- Evaluated model performance using **accuracy, precision, recall, F1-score**, and **ROC-AUC** metrics.  

#### **Model Performance**
- **Training Accuracy:** 100%  
- **Test Accuracy:** 99.76%  
- **Cross-Validation Accuracy:** 99.54%  
- **AUC:** 1.00  
- Only **8 misclassifications out of 3,356 samples** — excellent performance.  

---

## 💡 Key Insights

| **Category** | **Observation** |
|---------------|----------------|
| **Total Samples** | 16,776 |
| **Safe Samples** | 4,743 (28%) |
| **Unsafe Samples** | 12,033 (72%) |
| **Top Unsafe Parameters** | EC, NO₃, HCO₃ |
| **Top Unsafe States** | Maharashtra, Andhra Pradesh, Telangana, Rajasthan, Tamil Nadu, Haryana |
| **Major Correlations** | EC ↔ Na ↔ Cl ↔ Total Hardness |
| **Dominant Issue** | Salinity and mineral contamination |

---

## 📈 Visual Outputs

| **Visualization** | **Description** |
|--------------------|-----------------|
| **pH Distribution Plot** | KDE plot showing the distribution of pH values. |
| **Unsafe States Plot** | Top 10 states with the most unsafe groundwater samples. |
| **Correlation Heatmap** | Correlation among groundwater quality parameters. |
| **Confusion Matrix** | Random Forest confusion matrix showing classification accuracy. |
| **ROC Curve** | ROC curve illustrating near-perfect classification performance. |

---

## 🌎 Geospatial Visualization
The **map.html** file displays groundwater safety visually using an **interactive Folium map**.

- 🟢 **Green markers:** Safe water samples  
- 🔴 **Red markers:** Unsafe water samples  

Unsafe clusters are concentrated in **Gujarat, Rajasthan, Maharashtra, Tamil Nadu, and Andhra Pradesh**,  
while **eastern India** shows relatively safer groundwater conditions.

---

## 🧠 Future Enhancements

To further expand and refine this project:
- Add **Feature Importance** and **SHAP Explainability** for model interpretability.  
- Create **Choropleth maps** to show state/district unsafe percentages.  
- Include **Temporal Analysis** (if `Year` column available).  
- Develop a **Water Quality Index (WQI)** for risk grading.  
- Deploy the project using **Streamlit** or **Dash** for real-time interactive dashboards.  

---

## 🧰 Tech Stack

| **Category** | **Tools Used** |
|---------------|----------------|
| **Programming Language** | Python 3.11+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, folium |
| **Machine Learning** | scikit-learn |
| **Notebook Environment** | Jupyter / Google Colab |

---

## 📜 License
This project is licensed under the **MIT License**.  
You may use, modify, and distribute this project freely with proper credit.

---

## 👩‍💻 Author
**Asmility / Asminity**  
📧 [asmit12yadav@gmail.com](mailto:asmit12yadav@gmail.com)  
🌐 [https://github.com/asminity](https://github.com/asminity)

> *“Sustainable groundwater management starts with understanding data.”*

