import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import io
import base64

# Konfigurasi halaman
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🎯",
    layout="wide"
)

# Inisialisasi MediaPipe
@st.cache_resource
def load_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )
    return face_detection, mp_drawing, mp_face_detection

def detect_faces_in_image(image, confidence_threshold=0.5):
    """Deteksi wajah dalam gambar"""
    face_detection, mp_drawing, mp_face_detection = load_face_detection()
    
    # Konversi PIL ke numpy array
    img_array = np.array(image)
    
    # Pastikan format RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Deteksi wajah
    results = face_detection.process(img_array)
    
    # Gambar hasil deteksi
    annotated_image = img_array.copy()
    face_count = 0
    
    if results.detections:
        face_count = len(results.detections)
        for detection in results.detections:
            if detection.score[0] > confidence_threshold:
                mp_drawing.draw_detection(annotated_image, detection)
    
    return annotated_image, face_count, results.detections

def main():
    # Header
    st.title("🎯 Face Detection App")
    st.markdown("Upload gambar untuk mendeteksi wajah menggunakan MediaPipe")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan")
    confidence = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Threshold untuk deteksi wajah (semakin tinggi semakin ketat)"
    )
    
    # Informasi di sidebar
    st.sidebar.markdown("### 📋 Informasi")
    st.sidebar.info(
        "Aplikasi ini menggunakan MediaPipe Google untuk deteksi wajah. "
        "Upload gambar dengan format JPG, JPEG, atau PNG."
    )
    
    # Statistik
    st.sidebar.markdown("### 📊 Statistik")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk dianalisis:",
            type=['jpg', 'jpeg', 'png'],
            help="Maksimal ukuran file 200MB"
        )
        
        # Contoh gambar
        if st.button("🎲 Coba dengan Gambar Demo"):
            # Buat gambar demo sederhana
            demo_img = Image.new('RGB', (400, 300), color='lightblue')
            st.session_state['demo_mode'] = True
            st.session_state['demo_img'] = demo_img
    
    with col2:
        st.header("🔍 Hasil Deteksi")
        
        # Placeholder untuk hasil
        result_placeholder = st.empty()
        
        # Proses gambar yang diupload
        if uploaded_file is not None:
            try:
                # Baca gambar
                image = Image.open(uploaded_file)
                
                # Tampilkan gambar asli
                st.subheader("Gambar Asli")
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
                
                # Deteksi wajah
                with st.spinner("🔍 Mendeteksi wajah..."):
                    annotated_img, face_count, detections = detect_faces_in_image(image, confidence)
                
                # Tampilkan hasil
                st.subheader("Hasil Deteksi")
                st.image(annotated_img, caption=f"Terdeteksi {face_count} wajah", use_column_width=True)
                
                # Statistik deteksi
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Jumlah Wajah", face_count)
                with col_stat2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col_stat3:
                    st.metric("Status", "✅ Berhasil" if face_count > 0 else "❌ Tidak ada wajah")
                
                # Detail deteksi
                if detections:
                    st.subheader("📋 Detail Deteksi")
                    for i, detection in enumerate(detections):
                        if detection.score[0] > confidence:
                            st.write(f"**Wajah {i+1}:** Confidence Score = {detection.score[0]:.3f}")
                
                # Tombol download hasil
                if face_count > 0:
                    # Konversi hasil ke bytes untuk download
                    result_pil = Image.fromarray(annotated_img.astype('uint8'))
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="💾 Download Hasil",
                        data=buf.getvalue(),
                        file_name="face_detection_result.png",
                        mime="image/png"
                    )
                        
            except Exception as e:
                st.error(f"❌ Error saat memproses gambar: {str(e)}")
        
        # Demo mode
        elif st.session_state.get('demo_mode', False):
            st.info("🎲 Mode demo aktif - Upload gambar nyata untuk deteksi wajah")
    
    # Footer dengan informasi
    st.markdown("---")
    
    # Tabs untuk informasi tambahan
    tab1, tab2, tab3 = st.tabs(["📖 Cara Penggunaan", "🔧 Teknologi", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ### 📖 Cara Menggunakan Aplikasi
        
        1. **Upload Gambar**: Klik tombol "Browse files" dan pilih gambar
        2. **Atur Confidence**: Gunakan slider di sidebar untuk mengatur threshold
        3. **Lihat Hasil**: Hasil deteksi akan muncul secara otomatis
        4. **Download**: Klik tombol download untuk menyimpan hasil
        
        ### 💡 Tips untuk Hasil Terbaik
        - Gunakan gambar dengan pencahayaan yang baik
        - Pastikan wajah terlihat jelas dan tidak terlalu kecil
        - Format gambar yang didukung: JPG, JPEG, PNG
        - Confidence 0.5 - 0.7 biasanya memberikan hasil optimal
        """)
    
    with tab2:
        st.markdown("""
        ### 🔧 Teknologi yang Digunakan
        
        - **MediaPipe**: Framework machine learning Google untuk deteksi wajah
        - **Streamlit**: Framework untuk membuat web app Python
        - **PIL/Pillow**: Library untuk pemrosesan gambar
        - **NumPy**: Library untuk komputasi numerik
        
        ### ⚡ Keunggulan MediaPipe
        - Ringan dan cepat
        - Akurasi tinggi
        - Tidak memerlukan GPU
        - Optimized untuk mobile dan web
        """)
    
    with tab3:
        st.markdown("""
        ### ❓ Frequently Asked Questions
        
        **Q: Kenapa wajah tidak terdeteksi?**
        A: Coba turunkan confidence threshold atau pastikan gambar memiliki pencahayaan yang baik.
        
        **Q: Apakah data saya aman?**
        A: Ya, semua pemrosesan dilakukan secara lokal dan gambar tidak disimpan.
        
        **Q: Format gambar apa yang didukung?**
        A: JPG, JPEG, dan PNG dengan ukuran maksimal 200MB.
        
        **Q: Bisakah mendeteksi wajah yang miring?**
        A: Ya, MediaPipe dapat mendeteksi wajah dengan berbagai orientasi.
        """)
    
    # Sidebar statistics update
    st.sidebar.markdown("### 📈 Session Stats")
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = 0
    if 'total_faces' not in st.session_state:
        st.session_state.total_faces = 0
    
    if uploaded_file is not None:
        st.session_state.processed_images += 1
        if 'face_count' in locals():
            st.session_state.total_faces += face_count
    
    st.sidebar.metric("Gambar Diproses", st.session_state.processed_images)
    st.sidebar.metric("Total Wajah", st.session_state.total_faces)

if __name__ == "__main__":
    main()
