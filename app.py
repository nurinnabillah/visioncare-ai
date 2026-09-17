import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt
import time
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from skimage.measure import shannon_entropy
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
if "history" not in st.session_state:
    st.session_state.history = []

if "force_accept" not in st.session_state:
    st.session_state.force_accept = False

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# Hide Streamlit UI elements (Deploy bar, menu, footer)
# st.markdown("""
#     <style>
#     header {visibility: hidden;}
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

st.set_page_config(
    page_title="Vision Care AI",
    page_icon="🩺",  
    layout="centered"
)

st.set_page_config(initial_sidebar_state="expanded")

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Footer
def show_footer():
    st.markdown("""
    <div class="custom-footer">
        <p>Developed by <b>Nurin Nabilah Rosidi</b> | Final Year Project 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")   # css

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "final_densenet169_progressive_focal.keras",  
        compile=False
    )
    return model

model = load_model()

# To check user upload retina images or not
def is_likely_retina(img):
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # 1. Enhance textures
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 2. INCREASE MASK SIZE
    # We increase the circle to 120 so that large retinas aren't cut off
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 120, 255, -1) 
    
    # 3. Detect Edges
    edges = cv2.Canny(enhanced_gray, 30, 150)
    
    # 4. MORE FLEXIBLE RATIO
    inner_detail = np.sum(cv2.bitwise_and(edges, edges, mask=mask) > 0)
    outer_detail = np.sum(cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(mask)) > 0)
    
    # We change 0.5 to 0.8. This allows clear retina edges to pass, 
    # but still blocks human photos where edges are everywhere.
    if outer_detail > (inner_detail * 0.4): 
        return False 

    # 5. Existing Straight Line Check (for Buttons)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=10)
    straight_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 2 or abs(y1 - y2) < 2:
                straight_lines += 1
    if straight_lines > 12:
        return False 

    # 6. Final Decision
    edge_density = np.sum(edges > 0) / (256 * 256)
    
    # LOWER DENSITY THRESHOLD
    # Clear retinas like your last one might have very clean vessels. 
    # 0.03 is safer than 0.04.
    if edge_density > 0.05:
        return True 
    
    # 7. Overexposed/White Retina Check (Img 5 / Img 24)
    
    corners_val = [gray[0,0], gray[0,255], gray[255,0], gray[255,255]]
    if np.mean(gray) > 170 and np.mean(corners_val) < 215:
        return True 

    return False

# Background Image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: 
                linear-gradient(135deg, rgba(20,3,60,0.85), rgba(28,6,61,0.85)),
                url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/wallpaper.png")

# Class labels ikut dataset
class_labels = ["Mild", "Moderate", "No Diabetic Retinopathy", "Proliferative", "Severe"]
graph_labels = ["Mild", "Moderate", "No_DR", "Proliferative", "Severe"]

# CLAHE Function
def apply_clahe(img_pil):
    """Apply CLAHE preprocess ke retina image"""
    img = np.array(img_pil)

    # Convert ke LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE pada L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge balik
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return Image.fromarray(final)

# Resize and Normalize Functions 
def preprocess_image(img):
    """Resize + normalize"""
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Confidence Text
def generate_confidence_text(preds):
    sorted_probs = np.sort(preds[0])[::-1]
    top1, top2 = sorted_probs[0], sorted_probs[1]

    # Confidence gap
    gap = top1 - top2

    if gap < 0.10:
        return "⚠️ The model is unsure — two classes are very close."
    elif gap < 0.20:
        return "🟡 Moderate confidence — prediction is likely but still uncertain."
    else:
        return "🟢 High confidence — prediction is clearly separated from other classes."

# Grad_CAM
def get_gradcam(model, img_array, class_index, layer_name, eps=1e-8):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    img_array = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_index]

    # Formula
    grads = tape.gradient(loss, conv_outputs)
    # safe replacement for NaN/Inf
    grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
    grads = tf.where(tf.math.is_inf(grads), tf.zeros_like(grads), grads)

    # Global-average-pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)
    conv_outputs = conv_outputs[0]  # (h, w, channels)

    # Weighted sum of conv_outputs channels
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Convert to numpy and relu
    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)

    # Normalize to 0..1 safely
    denom = (np.max(heatmap) - np.min(heatmap)) + eps
    heatmap = (heatmap - np.min(heatmap)) / denom

    return heatmap.astype(np.float32)

# Heatmap for Grad-CAM
def postprocess_heatmap(heatmap, target_size, blur_ksize=(5,5), apply_blur=True):

    # target_size: (width, height)
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    if apply_blur:
        # kernel size must be odd
        kx, ky = blur_ksize
        if kx % 2 == 0: kx += 1
        if ky % 2 == 0: ky += 1
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (kx, ky), 0)

    # Normalize again to 0..1
    if np.max(heatmap_resized) > 0:
        heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    else:
        heatmap_norm = heatmap_resized

    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    return heatmap_norm.astype(np.float32), heatmap_uint8

# Detect important location at heatmap
def get_bounding_boxes_from_heatmap(heatmap_uint8, method="fixed", fixed_thresh=0.75, min_area=500, morph_k=2):

    if method == "adaptive":
        # Adaptive threshold expects a grayscale image
        thr = cv2.adaptiveThreshold(
            heatmap_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        ret, thr = cv2.threshold(heatmap_uint8, int(fixed_thresh * 255), 255, cv2.THRESH_BINARY)

    # Morphological opening to remove tiny noisy blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    # thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)  # close small gaps

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

    # Optionally sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    return boxes, thr  # thr returned for debugging/visual checks

# Show the bounding boxes
def overlay_gradcam_and_draw_boxes(original_img_pil, heatmap_norm, heatmap_uint8, boxes, alpha=0.35, box_color=(255,255,255)):

    original_img = np.array(original_img_pil).astype(np.uint8)
    h, w = original_img.shape[:2]

    # Create colored heatmap (apply COLORMAP_JET to heatmap_uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Combine with original: convert to float for blending
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    annotated = superimposed.copy()

    # Draw boxes and labels
    for (x, y, bw, bh) in boxes:
        # ensure coordinates inside image
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w-1, x + bw), min(h-1, y + bh)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        # label background
        label = "AI Focus"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # draw filled rect for label
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), box_color, -1)
        # put label in contrasting color
        cv2.putText(annotated, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

    # Convert to RGB (if needed) and return
    return annotated

# ---------- End helper functions ----------

# ---------- PDF GENERATION FUNCTION ----------
def safe_text(text):
    if text is None:
        return ""
    return str(text).encode("latin-1", "ignore").decode("latin-1")

def generate_pdf(prediction, confidence, history, gradcam_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== HEADER =====
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "AI Retinal Screening Report", ln=True, align="C")

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100,100,100)
    pdf.cell(0, 6, "Educational screening only - Not a medical diagnosis", ln=True, align="C")
    pdf.ln(4)

    pdf.set_draw_color(180,180,180)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(10)

    # ===== PREDICTION SUMMARY BOX =====
    pdf.set_fill_color(245, 248, 250)
    pdf.set_draw_color(200, 200, 200)

    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 8, "Prediction Summary", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Detected Stage : {prediction}", ln=True)
    pdf.cell(0, 6, f"Confidence Score : {confidence:.2f}%", ln=True)
    pdf.ln(10)

    # ===== GRAD-CAM =====
    if gradcam_path is not None:
        pdf.set_draw_color(200,200,200)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(8)

        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Grad-CAM Visualization", ln=True)

        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(80,80,80)
        pdf.multi_cell(
            0, 6,
            "Highlighted regions represent areas that most influenced the AI model's prediction."
        )
        pdf.ln(3)

        # Center image
        img_width = 85
        x_pos = (pdf.w - img_width) / 2
        pdf.image(gradcam_path, x=x_pos, w=img_width)

        pdf.ln(4)
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(120,120,120)
        pdf.cell(0, 6, "Figure: Grad-CAM heatmap with detected focus regions", ln=True, align="C")
        pdf.ln(6)

    # ===== PAST RECORDS =====
    pdf.set_draw_color(200,200,200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 8, "Past Screening Records", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(50,50,50)

    if len(history) == 0:
        pdf.cell(0, 6, "No past records found.", ln=True)
    else:
        for i, item in enumerate(history[-3:], start=1):
            pdf.cell(
                0, 6,
                f"{i}. {item['time']} - {item['prediction']} ({item['confidence']:.1f}%)",
                ln=True
            )

    pdf.ln(10)

    # ===== DISCLAIMER =====
    pdf.set_draw_color(200,200,200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(160, 0, 0)
    pdf.cell(0, 8, "Disclaimer", ln=True)

    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0, 6,
        "This AI system provides an automated screening result and may not always be accurate. "
        "The report is intended for educational and preliminary screening purposes only and "
        "must not be considered a medical diagnosis."
    )

    return pdf.output(dest="S").encode("latin-1")

# -------- Sidebar Navigation -----------------
home_icon = get_base64_image("images/home.png")
upload_icon = get_base64_image("images/upload.png")
about_icon = get_base64_image("images/about.png")

st.markdown(f"""
    <style>
        /* Stylize Logo */
        [data-testid="stSidebar"] img {{
            filter: brightness(0) invert(1);
            width: 80% !important;
            margin: 20px auto;
            display: block;
        }}

        /* Styling Butang Sidebar */
        div[data-testid="stSidebar"] button {{
            background-color: #262730; /* Warna gelap asal */
            color: white !important;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 10px 20px;
            height: 50px;
            text-align: left;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }}

        /* Hover Effect */
        div[data-testid="stSidebar"] button:hover {{
            background-color: #5e3a8c !important; /* Ungu hover */
            border-color: #7b4db3;
            transform: translateX(5px);
        }}

        /* Icon setup menggunakan background image (Base64) */
        div[data-testid="stSidebar"] button p::before {{
            content: "";
            display: inline-block;
            width: 25px;
            height: 25px;
            margin-right: 15px;
            background-size: contain;
            background-repeat: no-repeat;
            vertical-align: middle;
            filter: brightness(0) invert(1); /* Biar icon jadi putih */
        }}

        /* Assign icon spesifik ikut button ID/Key */
        div[data-testid="stSidebar"] button[key="home"] p::before {{ background-image: url('data:image/png;base64,{home_icon}'); }}
        div[data-testid="stSidebar"] button[key="upload"] p::before {{ background-image: url('data:image/png;base64,{upload_icon}'); }}
        div[data-testid="stSidebar"] button[key="about"] p::before {{ background-image: url('data:image/png;base64,{about_icon}'); }}

    </style>
""", unsafe_allow_html=True)

st.sidebar.image("images/logo.png", use_container_width=True)

# Define pages
pages = {
    "🏠 Home": "home",
    "📤 Upload Image": "upload",
    "📊 About System": "about",
    # "🛠️ Check Layers": "layers"
}

if "history" not in st.session_state:
    st.session_state.history = []

# State untuk simpan page aktif
if "page" not in st.session_state:
    st.session_state.page = "home"

# Render buttons
for name, key in pages.items():
    if st.sidebar.button(name, key=key, use_container_width=True):
        st.session_state.page = key

# ----------- Page 1: Home ---------------
if st.session_state.page == "home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="
              color: #00FFA3; 
                font-size: 3em; 
                margin-bottom: 0px;
                text-shadow: 0px 4px 10px rgba(0, 255, 163, 0.2);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                Vision Care AI System
            </h1>
            <p style="
                color: #D3D3D3; 
                font-weight: bold; 
                font-size: 1.1em; 
                margin-top: 15px; 
                margin-bottom: 20px;
                letter-spacing: 0.5px;
            ">
                Automated Classification Diabetic Retinopathy System
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div style="text-align: center; padding: 0 20px;">
            <p>Welcome! This tool provides a <b>quick and supportive</b> way to screen your <b>retina images</b> 
            for early signs of Diabetic Retinopathy (DR) using a high-accuracy Deep Learning model.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. FEATURE CARDS (Highlighting Key Benefits)
    st.subheader("💡 Why Choose This System?")
    col_feat_1, col_feat_2, col_feat_3 = st.columns(3)
    
    with col_feat_1:
        st.markdown("""
        <div class='feature-container'>
        <div class='feature-card purple'>
            <h4>⚡ Instant Analysis</h4>
            <p>Get results in seconds. No complex steps required.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat_2:
        st.markdown("""
        <div class='feature-card yellow'>
            <h4>🧠 AI Focus </h4>
            <p>See exactly where the AI detects signs of disease.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat_3:
        st.markdown("""
        <div class='feature-card blue'>
            <h4>✅ High Accuracy</h4>
            <p>Built on DenseNet169 and advanced preprocessing.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. CORE EXPLANATION (Refined Card)
    st.markdown("""
        <div class="card" style="background:#2d1b4e; color:white; padding: 25px; border-radius: 12px; border: 1px solid #8e44ad;">
            <h3 style='color:#a67bbd; margin-top:0;'>🔬 Understanding Diabetic Retinopathy</h3>
            <p style = 'text-align: justify;'>
            Diabetic Retinopathy occurs when high blood sugar damages the blood vessels in the retina. 
            Early detection is crucial, as prompt treatment can protect your vision from becoming 
            blurry or permanently blind. This system gives you an efficient first insight into your eye condition. 
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # 4. CALL TO ACTION & IMAGE
    st.subheader("🖼️ Retina Image Analysis")
    # Use 3 columns to display a more engaging example
    img_col, button_col = st.columns([2, 1])

    with img_col:
         # Replace with a more engaging example image or CLAHE demonstration (if available)
         # I am using a placeholder URL here. If you have the uploaded image file, use that name.
         st.image("images/example_eyes.jpg", caption="Retina Image", width=400)
    
    with button_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.success("Ready to analyze your image? Just one click!")
        
        # Larger and more prominent button
        if st.button("🚀 START ANALYSIS", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    show_footer()
   
# ----------- Page 2: Upload Image ---------------
elif st.session_state.page == "upload":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📤 Upload Retina Image")

    # Step Indicator
    st.markdown("""
    <div class="step-container">
        <div class="step active">1️⃣ Upload</div>
        <div class="step">2️⃣ Predict</div>
        <div class="step">3️⃣ View Result</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # If user uploads a different image, reset checkbox
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.force_accept = False
            st.session_state.last_uploaded_file = uploaded_file.name

    if uploaded_file is not None:
        img_format = Image.open(uploaded_file).format
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="Preview of Uploaded Image", width=400)

       # 1. Run your existing detection function
        is_retina = is_likely_retina(image)

        if not is_retina:
            # Show a warning instead of an immediate error
            st.warning("⚠️ The system could not automatically verify this as a retinal image (it may be too faded or overexposed).")
            
            # Provide a checkbox for the user to manually confirm
            force_accept = st.checkbox(
                "I confirm that this is a valid retinal fundus image.",
                key="force_accept"
            )
            
            if not force_accept:
                # If it's not detected AND the user hasn't checked the box, stop here
                st.error("❌ Invalid image detected. Please upload a valid retinal image or check the box above to proceed manually.")
                st.stop()
            else:
                # If the user checks the box, show a note and allow the code to continue
                st.info("💡 Proceeding with Manual Override. Please ensure the image is correct for accurate analysis.")

        # If the code reaches here, it means (is_retina is True) OR (force_accept is True)
        st.success("✅ Image accepted for analysis.")

        st.markdown("### 📃 About Uploaded Image")
        img_info_col1, img_info_col2 = st.columns(2)

        with img_info_col1:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Format:** {img_format if img_format else 'N/A'}")
        
        with img_info_col2:
            st.write(f"**Dimensions:** {image.width} x {image.height} px")
            st.write(f"**Upload Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.info("Click 'Predict' below to analyze this image using the AI model.")

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image... please wait..."):
                time.sleep(2)

                # CLAHE preprocess
                image_clahe = apply_clahe(image)

                # Show original & preprocessed
                st.subheader("Original vs CLAHE Preprocessed")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    st.image(image_clahe, caption="CLAHE Preprocessed", use_container_width=True)

                # Preprocess & predict
                img_preprocessed = preprocess_image(image_clahe)

                preds = model.predict(img_preprocessed, verbose=0) 
                predicted_class = class_labels[np.argmax(preds)]
                confidence = np.max(preds) * 100

                st.markdown("""
                    <div class="step-container">
                    <div class="step">1️⃣ Upload</div>
                    <div class="step active">2️⃣ Predict</div>
                    <div class="step">3️⃣ View Result</div>
                </div>
                """, unsafe_allow_html=True)

                # Color-coded prediction result box
                color_map = {
                    "No_DR": "#3498db",        # blue
                    "Mild": "#f1c40f",         # yellow
                    "Moderate": "#e67e22",     # orange
                    "Severe": "#e74c3c",       # red
                    "Proliferative": "#8e44ad" # purple
                }
                box_color = color_map.get(predicted_class, "#ffffff")

                st.markdown(f"""
                    <div class="result-box" style="
                        border-left: 8px solid {box_color};">
                        <h4>🧠 Prediction Result:</h4>
                        <p><b>Detected Stage:</b> 
                            <span style="color:{box_color}; font-weight:600;">{predicted_class}</span></p>
                        <p><b>Confidence:</b> {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability Bar Chart
                st.subheader("Prediction Probability")
                fig, ax = plt.subplots()
                ax.bar(graph_labels, preds[0], color=[color_map.get(lbl, "#95a5a6") for lbl in graph_labels])
                ax.set_ylabel("Probability")
                ax.set_title("Model Confidence per Class")
                st.pyplot(fig)

                st.markdown("""
                    <div class="step-container">
                    <div class="step">1️⃣ Upload</div>
                    <div class="step">2️⃣ Predict</div>
                    <div class="step active">3️⃣ View Result</div>
                </div>
                """, unsafe_allow_html=True)

                # After preds calculated:
                predicted_index = np.argmax(preds)
                predicted_class = class_labels[predicted_index]
                confidence = np.max(preds) * 100

                # Confidence Explanation 
                conf_text = generate_confidence_text(preds)
                st.info(f"**Model Confidence Note:** {conf_text}")

                layer_for_gradcam = "conv5_block32_concat"  

                # 1) compute heatmap (small spatial map)
                heatmap_small = get_gradcam(model, img_preprocessed, predicted_index, layer_name=layer_for_gradcam)

                # 2) resize + blur + normalize to original image size
                target_size = (image_clahe.width, image_clahe.height)  # (w, h)
                heatmap_norm, heatmap_uint8 = postprocess_heatmap(heatmap_small, target_size, blur_ksize=(3,3), apply_blur=False)

                # 3) find bounding boxes (adaptive or fixed)
                if predicted_class != "No Diabetic Retinopathy":
                    boxes, thr_mask = get_bounding_boxes_from_heatmap(
                        heatmap_uint8,
                        method="fixed",
                        fixed_thresh=0.75,
                        min_area=40,
                        morph_k=3
                    )
                else:
                    boxes = []
                    thr_mask = None

                # 4) overlay + draw boxes
                if predicted_class == "No Diabetic Retinopathy":
                    annotated = image_clahe
                else:
                    annotated = overlay_gradcam_and_draw_boxes(image_clahe,heatmap_norm,heatmap_uint8,boxes,alpha=0.35)

               # 5) show results 
                if predicted_class != "No Diabetic Retinopathy":
                    if len(boxes) > 0:
                        st.success(f"Grad-CAM + AI focus zones detected: {len(boxes)} box(es).")
                    else:
                        st.warning("Model suggests lesion but no strong localized hotspots found (boxes empty).")
                else:
                    st.info("Predicted: Healthy retina — no bounding boxes drawn.")

                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.image(annotated, caption=f"Grad-CAM ({layer_for_gradcam}) + AI Focus", width=400)

                # === SAVE GRAD-CAM IMAGE FOR PDF ===
                gradcam_path = "gradcam_result.png"

                if isinstance(annotated, Image.Image):
                    annotated.save(gradcam_path)
                else:
                    Image.fromarray(annotated).save(gradcam_path)

                # === Save History Entry ===
                st.session_state.history.append({
                    "filename": uploaded_file.name,
                    "prediction": predicted_class,
                    "confidence": float(confidence),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                advice_text = ""
                if predicted_class == "Proliferative":
                    advice_text = (
                        "WARNING: Signs of proliferative diabetic retinopathy detected. "
                        "You should urgently consult an eye specialist."
                    )
                elif predicted_class == "Severe":
                    advice_text = (
                        "WARNING: Severe nonproliferative diabetic retinopathy detected. "
                        "Please schedule an appointment with an ophthalmologist within one month."
                    )
                elif predicted_class == "Moderate":
                    advice_text = (
                        "Moderate nonproliferative diabetic retinopathy detected. "
                        "You should arrange a detailed eye examination within 3 to 6 months."
                    )
                elif predicted_class == "Mild":
                    advice_text = (
                        "Mild nonproliferative diabetic retinopathy detected. "
                        "Schedule regular eye examinations every 6–12 months."
                    )
                else:
                    advice_text = (
                        "No signs of diabetic retinopathy detected. Maintain yearly screenings."
                    )

                if advice_text:
                    st.warning(advice_text)

                # DOWNLOAD PDF BUTTON 
                
                pdf_bytes = generate_pdf(
                    predicted_class,
                    confidence,
                    st.session_state.history,
                    gradcam_path=gradcam_path
                )

                st.markdown('<div class = "btn-report">', unsafe_allow_html = True)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="DR REPORT.pdf",
                    mime="application/pdf"
                )

                st.markdown('</div>', unsafe_allow_html = True)

    # Footer
    show_footer()

# -------- Page 3: About System --------------

elif st.session_state.page == "about":
    # The 'card' class handles the main container background and padding
    st.markdown("<div class='card'>", unsafe_allow_html=True) 
    st.title("ℹ️ About This System")

    # 1. OVERVIEW (Using st.info for a prominent block)
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(82, 0, 204, 0.9), rgba(0, 153, 255, 0.8));
            border-left: 8px solid #a020f0;
            border-radius: 16px;
            padding: 25px;
            color: #ffffff;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
            font-size: 16px;
            line-height: 1.6;
        ">
        <h3 style="color:#ffddff; margin-bottom:10px;">🧠 Overview: AI-Powered DR Screening</h3>

        The <b>Automated Diabetic Retinopathy Detection System</b> is a cutting-edge deep learning application 
        designed for the <b>early screening of Diabetic Retinopathy (DR)</b>. DR is a diabetes-related eye disease 
        that causes blindness if left undiagnosed and untreated.

        Our system utilizes the <b>DenseNet-169 architecture</b> (a Convolutional Neural Network or CNN), refined 
        through <b>transfer learning</b>, <b>progressive layer unfreezing</b>, and a <b>custom focal loss function</b> 
        to ensure high-accuracy classification across all five stages of the disease, even with data imbalances.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. HOW IT WORKS (Using Columns for a step-by-step visual flow)
    st.subheader("⚙️ How the Prediction Works")
    
    col_upload, col_clahe, col_predict, col_gradcam = st.columns(4)

    with col_upload:
        st.markdown(
            """
            <div style='background-color: #3b2c50; padding: 10px; border-radius: 8px; text-align: center;'>
                <p style='color: #8e44ad; font-size:1.1em; font-weight: 700;'>1. Image Upload</p>
                <p style='font-size: 0.9em;'>User uploads the fundus (retinal) image.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_clahe:
        st.markdown(
            """
            <div style='background-color: #3b2c50; padding: 10px; border-radius: 8px; text-align: center;'>
                 <p style='color: #f1c40f; font-size:1.1em; font-weight: 700;'>2. Preprocessing</p>
                <p style='font-size: 0.9em;'>CLAHE applied to enhance clarity and contrast.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_predict:
        st.markdown(
            """
            <div style='background-color: #3b2c50; padding: 10px; border-radius: 8px; text-align: center;'>
                 <p style='color: #2ecc71; font-size:1.1em; font-weight: 700;'>3. Model Prediction</p>
                <p style='font-size: 0.9em;'>DenseNet-169 classifies the image into one of 5 stages.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_gradcam:
        st.markdown(
            """
            <div style='background-color: #3b2c50; padding: 10px; border-radius: 8px; text-align: center;'>
                 <p style='color: #3498db; font-size:1.1em; font-weight: 700;'>4. Visualization</h4>
                <p style='font-size: 0.9em;'>Grad-CAM heatmap shows the AI's area of focus.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")


    # 3. MODEL INFORMATION & SYSTEM BENEFITS (Side-by-Side Comparison)
    st.subheader("📊 Technical Details & Key Benefits")

    col_tech, col_benefits = st.columns(2)

    with col_tech:
        st.markdown("""
        <div class="card" style="background:#1b1b2f; padding: 20px; border-left: 5px solid #a67bbd;">
            <h4 style='color: #a67bbd; margin-top:0;'>⚙️ Model Specifications</h4>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li><b style='color:#f1c40f;'>Model:</b> DenseNet169 (Transfer Learning)</li>
                <li><b style='color:#f1c40f;'>Framework:</b> TensorFlow / Keras</li>
                <li><b style='color:#f1c40f;'>Accuracy:</b> ~90% on test data</li>
                <li><b style='color:#f1c40f;'>Enhancements:</b> CLAHE, Focal Loss, Progressive Unfreezing</li>
                <li><b style='color:#f1c40f;'>Dataset:</b> EyePACS, APTOS, Messidor combined</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_benefits:
        st.markdown("""
        <div class="card" style="background:#1b1b2f; padding: 20px; border-left: 5px solid #2ecc71;">
            <h4 style='color: #2ecc71; margin-top:0;'>✅ Advantages</h4>
            <ul style='padding-left: 20px;'>
                <li>Fast & Non-Invasive initial screening.</li>
                <li>Supports Clinicians by highlighting potential issues.</li>
                <li>Reduces Workload with AI-assisted interpretation.</li>
                <li>Ideal for Telemedicine and remote diagnosis settings.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # 4. PROJECT DETAILS (Footer - New Title)
    st.subheader("🎓 Academic Project Context") 
    
    col_project, col_supervisor = st.columns(2)
    
    with col_project:
        st.markdown(f"**Project Title:** Automated Detection Diabetic Retinopathy In Retina Images Using Deep Learning")
        st.markdown(f"**Student:** Nurin Nabilah binti Rosidi")

    with col_supervisor:
        st.markdown(f"**Supervisor:** Madam Hajar Izzati binti Mohd Ghazalli")
        st.markdown(f"**Institution:** Universiti Teknologi Mara (UiTM)")

    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; padding: 10px; border-radius: 8px; background: rgba(130, 71, 194, 0.1);">
            <i style='color:#a67bbd;'>“Early detection saves vision — empowering patients and doctors through AI-driven insights.”</i>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    show_footer()

