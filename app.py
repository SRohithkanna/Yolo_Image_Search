import streamlit as st
import sys
import time
from pathlib import Path
from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts

from PIL import Image,ImageDraw,ImageFont
import base64
import json
import io

# streamlit run app.py
# Above code runs the application on port 8501

# streamlit run app.py --server.port 8080
# Above code runs the application on port 8080


# Add project root to the system path
sys.path.append(str(Path(__file__).parent))

def img_to_base64(image : Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def init_session_state():
    session_defaults = {
    "metadata" : None,
    "unique_classes" : [],
    "count_options" : {},
    "search_params" : {
        "search_mode" : "Any of selected classes(OR)",
        "selected_classes" : [],
        "thresholds" : {}
    },
    "search_results" : [],
    "show_boxes" : True,
    "grid_columns" : 3,
    "highlight_matches" : True
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Computer Vision Powered Search Application")

# Custom CSS for perfect grid layout
st.markdown(f"""
<style>
/* Main container adjustments */
.st-emotion-cache-1v0mbdj {{
    width: 100% !important;
    height: 100% !important;
}}

/* Column container - critical for grid layout */
.st-emotion-cache-1wrcr25 {{
    max-width: none !important;
    padding: 0 1rem !important;
}}

/* Individual column styling */
.st-emotion-cache-1n76uvr {{
    padding: 0.5rem !important;
}}

/* Image cards */
.image-card {{
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    margin-bottom: 20px;
    background: #f8f9fa;
}}

.image-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}}

.image-container {{
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
}}

.image-container img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.meta-overlay {{
    padding: 10px;
    background: rgba(0,0,0,0.85);
    color: white;
    font-size: 13px;
    line-height: 1.4;
}}
</style>
""", unsafe_allow_html=True)

# Main options
option = st.radio("Choose an option:",
                  ("Process a single image","Process image folder", "Load existing metadata"),
                  horizontal=True)

if option == "Process image folder":
    with st.expander("Process new image folder", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_dir = st.text_input("Image directory path:", placeholder="path/to/images")
        with col2:
            model_path = st.text_input("Model weights path:", "yolo11m.pt")

        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = YOLOv11Inference(model_path)
                        metadata = inferencer.process_directory(image_dir)
                        metadata_path = save_metadata(metadata, image_dir)
                        st.success(f"Processed {len(metadata)} images. Metadata saved to:")
                        st.code(str(metadata_path))
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning(f"Please enter an image directory path")


elif option == "Process a single image":
    with st.expander("Process a single image", expanded=True):
        image_path = st.text_input("Image file path:", placeholder="path/to/file.jpg")
        model_path = st.text_input("Model weights path:", "yolo11m.pt")

        if st.button("Run inference"):
            try:
                inferencer = YOLOv11Inference(model_path)
                result = inferencer.process_single(image_path)

                st.success("Inference complete!")

                # --- Draw boxes on the image ---
                img = Image.open(result["image_path"])
                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()

                for det in result["detections"]:
                    cls = det["class"]
                    bbox = det["bbox"]   # [x1, y1, x2, y2]
                    conf = det["confidence"]

                    # Style
                    color = "#FF4B4B"
                    thickness = 3

                    # Box
                    draw.rectangle(bbox, outline=color, width=thickness)

                    # Label
                    label = f"{cls} {conf:.2f}"
                    text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]

                    # Background rectangle above box
                    bg_rect = [
                        bbox[0],
                        bbox[1] - text_h - 4,
                        bbox[0] + text_w + 8,
                        bbox[1],
                    ]
                    draw.rectangle(bg_rect, fill=color)
                    draw.text(
                        (bbox[0] + 4, bbox[1] - text_h - 2),
                        label,
                        fill="white",
                        font=font,
                    )

                # Class counts summary text
                meta_items = [f"{k}: {v}" for k, v in result["class_counts"].items()]

                # --- Display card, same style as grid ---
                st.markdown(
                    f"""
                <div class="image-card">
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_to_base64(img)}">
                    </div>
                    <div class="meta-overlay">
                        <strong>{Path(result['image_path']).name}</strong><br>
                        {", ".join(meta_items) if meta_items else "No detections"}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Optional: raw JSON for debugging
                with st.expander("Raw detection metadata"):
                    st.json(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")


else :
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path = st.text_input("Metadata file path:", placeholder="path/to/matadata.json")

        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("Loading Metadata..."):
                        metadata = load_metadata(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        st.success(f"Successfully loaded metadata for {len(metadata)} images.")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")
            else:
                st.warning(f"Please enter a metadata file path")


                # Person, car, airplane, banana,apple
                # Person : 1,2,3,10

# st.write(f"{st.session_state.unique_classes}, {st.session_state.count_options}")

#Search functionality

if st.session_state.metadata:
    st.header("🔍 Search Engine")

 

    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Search mode:", 
                ("Any of selected classes (OR)", "All selected classes (AND)"),
                horizontal=True
        )

        st.session_state.search_params["selected_classes"] = st.multiselect(
            "Classes to search for:", 
            options=st.session_state.unique_classes
        )

        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Thresholds (optional)")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for {cls}",
                        options=["None"] + st.session_state.count_options[cls]
                    )

        if st.button("Search Images", type="primary") and st.session_state.search_params["selected_classes"]:
            results = []
            search_params = st.session_state.search_params

            for item in st.session_state.metadata:
                matches = False
                class_matches = {}

                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item['detections'] if d['class'] == cls]
                    class_count = len(class_detections)
                    # 10 person
                    class_matches[cls] = False

                    threshold = search_params["thresholds"].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = (class_count>=1)
                    else : 
                        class_matches[cls] = (class_count>=1 and class_count<= int(threshold))
                       

                if search_params["search_mode"] == "Any of selected classes (OR)":
                    # OR case
                    matches = any(class_matches.values())
                    
                else : # AND mode
                    
                    matches = all(class_matches.values())
                    
                
                if matches:
                    results.append(item)

            st.session_state.search_results = results

        # st.write(st.session_state.search_results)


if st.session_state.search_results:
    results=st.session_state.search_results 
    search_params=st.session_state.search_params

    st.subheader(f"RESULTS: {len(results)}  matching images ")

    #display controls
    with st.expander("Display options",expanded=True):
        cols=st.columns(3)
        with cols[0]:
            st.session_state.show_boxes=st.checkbox(
                "Show bounding boxes",
                value=st.session_state.show_boxes
            )
        with cols[1]:
            st.session_state.grid_columns=st.slider(
                "Grid column",
                min_value=2,
                max_value=6,
                value=st.session_state.grid_columns
            )
        with cols[2]:
            st.session_state.highlight_matches=st.checkbox(
                "Highlight matching classes",
                value=st.session_state.highlight_matches

            )

    grid_cols=st.columns(st.session_state.grid_columns)
    col_index=0

    for result in results:
        with grid_cols[col_index]:
            try:
                img=Image.open(result["image_path"])
                draw=ImageDraw.Draw(img)

                if st.session_state.show_boxes:
                    try:
                       font = ImageFont.truetype("arial.ttf",12)
                    except:
                        font= ImageFont.load_default()  
            
                    for det in result['detections']:
                        cls=det['class']
                        bbox=det['bbox']
                        color = "#15FF00"
                        thickness = 1

                        if cls in search_params["selected_classes"]:
                            color="#FF4B4B"
                            thickness = 3
                        elif st.session_state.highlight_matches:
                           continue
                        

                        draw.rectangle(bbox,outline=color,width=thickness)

                        if cls in  search_params["selected_classes"] or not st.session_state.highlight_matches:
                            label=f"{cls} {det["confidence"]: .2f}"
                            text_bbox=draw.textbbox((0,0),label,font=font)
                            text_width=text_bbox[2]-text_bbox[0]
                            text_width=text_bbox[3]-text_bbox[1]

                            draw.rectangle([bbox[0],bbox[1],bbox[0]+text_width+8,bbox[1]+text_width+4]
                                            ,outline=color)
                            draw.text(
                                (bbox[0]+4, bbox[1]+2),
                                label,
                                fill="white",
                                font=font
                            )
                
                meta_items = [f"{k}: {v}" for k, v in result['class_counts'].items() 
                                if k in search_params["selected_classes"]] 
                
                # Display card
                st.markdown(f"""
                <div class="image-card">
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_to_base64(img)}">
                    </div>
                    <div class="meta-overlay">
                        <strong>{Path(result['image_path']).name}</strong><br>
                        {", ".join(meta_items) if meta_items else "No matches"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error displaying {result["image_path"]} : {str(e)}")
        col_index = (col_index + 1) % st.session_state.grid_columns

    with st.expander("Export Options"):
        st.download_button(
            label="Download Results (JSON)",
            data = json.dumps(results,indent=2),
            file_name="search_results.json",
            mime="application/json"
        )
