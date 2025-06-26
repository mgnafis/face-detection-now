import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2

# Konfigurasi MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi RTC untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Konversi BGR ke RGB untuk MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        results = self.face_detection.process(rgb_img)
        
        # Gambar kotak di sekitar wajah yang terdeteksi
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🎯 Face Detection dengan Webcam")
    st.markdown("---")
    
    # Sidebar untuk pengaturan
    st.sidebar.header("⚙️ Pengaturan")
    detection_confidence = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    
    # Informasi aplikasi
    st.sidebar.markdown("### 📋 Informasi")
    st.sidebar.info(
        "Aplikasi ini menggunakan MediaPipe untuk deteksi wajah real-time melalui webcam."
    )
    
    # Tab untuk berbagai mode
    tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📸 Upload Image", "ℹ️ Info"])
    
    with tab1:
        st.header("Live Face Detection")
        st.markdown("Klik tombol **START** untuk memulai deteksi wajah real-time")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="face-detection",
            mode=webrtc_streamer.VideoMode.VIDEO_TRANSFORMER,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FaceDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Deteksi wajah sedang berjalan!")
        else:
            st.info("👆 Klik START untuk memulai deteksi")
    
    with tab2:
        st.header("Upload dan Deteksi Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Baca gambar
            image = Image.open(uploaded_file)
            
            # Konversi ke array numpy
            img_array = np.array(image)
            
            # Pastikan gambar dalam format RGB
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                # Inisialisasi MediaPipe
                with mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=detection_confidence
                ) as face_detection:
                    
                    # Deteksi wajah
                    results = face_detection.process(img_array)
                    
                    # Gambar hasil
                    annotated_image = img_array.copy()
                    
                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(annotated_image, detection)
                        
                        st.success(f"✅ Terdeteksi {len(results.detections)} wajah")
                    else:
                        st.warning("⚠️ Tidak ada wajah yang terdeteksi")
                    
                    # Tampilkan hasil
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Gambar Asli")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.subheader("Hasil Deteksi")
                        st.image(annotated_image, use_column_width=True)
    
    with tab3:
        st.header("Informasi Aplikasi")
        
        st.markdown("""
        ### 🔧 Teknologi yang Digunakan
        - **Streamlit**: Framework web app
        - **MediaPipe**: Library Google untuk deteksi wajah
        - **streamlit-webrtc**: Untuk akses webcam real-time
        - **PIL/Pillow**: Untuk pemrosesan gambar
        
        ### 📱 Fitur Utama
        - ✅ Deteksi wajah real-time melalui webcam
        - ✅ Upload dan analisis gambar
        - ✅ Pengaturan confidence threshold
        - ✅ Interface yang user-friendly
        
        ### 🚀 Cara Penggunaan
        1. **Live Detection**: Klik tab pertama dan tekan START
        2. **Upload Image**: Klik tab kedua dan upload gambar
        3. **Pengaturan**: Gunakan sidebar untuk mengatur confidence
        
        ### 💡 Tips
        - Pastikan pencahayaan cukup untuk deteksi optimal
        - Posisikan wajah menghadap kamera
        - Gunakan confidence threshold 0.5-0.7 untuk hasil terbaik
        """)
        
        # Status sistem
        st.markdown("### 🔍 Status Sistem")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MediaPipe", "✅ Ready")
        with col2:
            st.metric("Streamlit", "✅ Running")
        with col3:
            st.metric("WebRTC", "✅ Active")

if __name__ == "__main__":
    main()
