# 🔬 Blood Cell Detection using YOLOv10

![Blood Cell Detection](https://user-images.githubusercontent.com/your-image-placeholder.png)  

## 📌 Overview
This is a **Blood Cell Detection** web app built using **YOLOv10** and **Streamlit**.  
The model can detect and classify three types of blood cells:  
- 🟥 **RBC (Red Blood Cells)**  
- 🟩 **WBC (White Blood Cells)**  
- 🟨 **Platelets**  

🚀 **Live Demo:** [Click Here to Test](https://huggingface.co/spaces/rohanavhad/blood-cell-detection)  

---

## 🎯 **Features**
✅ **Upload an image** → Detects blood cells instantly.  
✅ **Bounding boxes** → Drawn around detected RBCs, WBCs, and Platelets.  
✅ **Displays class, confidence, and bounding box coordinates**.  
✅ **Shows precision, recall, and mAP scores** for model performance.  
✅ **User-friendly UI** with a clear structure.  

---

## 🖥️ **How to Use**
1️⃣ **Click on "Browse Files"** to upload an image of blood cells.  
2️⃣ **Wait for a few seconds** → The model will detect and classify the cells.  
3️⃣ **View results** → Bounding boxes will appear, and a table will show detection details.  
4️⃣ **Check performance** → A precision-recall table is provided for evaluation.  

---

## 📊 **Model Performance (YOLOv10)**
| Class      | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|------------|------------|------------|------------|
| Overall    | 0.85      | 0.86      | 0.91      | 0.63      |
| RBC        | 0.80      | 0.82      | 0.89      | 0.60      |
| WBC        | 0.88      | 0.85      | 0.92      | 0.65      |
| Platelets  | 0.75      | 0.78      | 0.87      | 0.58      |

**🔹 These scores were obtained from model validation using YOLOv10.**  

---

## 🔧 **Technology Stack**
- **Machine Learning Model:** YOLOv10  
- **Programming Language:** Python  
- **Frameworks:** Streamlit, Ultralytics  
- **Libraries Used:** OpenCV, Pandas, NumPy, Pillow  

---

## 🚀 **Deployment**
This app is hosted on **Hugging Face Spaces** using Streamlit.  


