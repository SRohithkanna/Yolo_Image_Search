## Computer Vision Powered Image Search Application

**object detection and visual search application** built using **YOLO** and **Streamlit**.  
The system detects objects in images, stores structured metadata, and allows users to **search images by object type and count** through an interactive web interface.

---

##  Project Overview

Searching large image datasets manually is inefficient.  
This project solves that problem by:

- Automatically detecting objects in images
- Extracting and storing detection metadata
- Enabling fast, flexible image search using object filters
- Visualizing results with annotated bounding boxes

The application supports **single-image inference**, **batch processing of image folders**, and **metadata reuse** without re-running inference.

---

##  Key Features

-  **Object Detection** using YOLO
-  **Batch Image Processing** from directories
-  **Single Image Inference** with visualization
-  **Search Engine**
  - OR / AND class-based filtering
  - Optional object count thresholds
-  **Metadata Storage** in JSON format
-  **Annotated Image Visualization**
-  **Fast Reload via Metadata Loading**
- **Session State Management** for smooth UI experience

---

##  Technologies Used

| Tool | Purpose |
|-----|--------|
| **Python** | Core programming language |
| **YOLO (Ultralytics)** | Object detection model |
| **Streamlit** | Web UI and application framework |
| **Pillow (PIL)** | Image processing and annotation |
| **JSON** | Metadata storage |
| **Base64** | Image embedding in UI |
| **CSS** | Custom UI styling |

---

## How to run this project
1.Clone the repository
git clone https://github.com/your-username/object-detection-streamlit.git](https://github.com/SRohithkanna/Yolo_Image_Search.git
---
cd object-detection-streamlit

2.Install requirements
---
pip install -r requirements.txt

3.Run the application
---
streamlit run app.py





