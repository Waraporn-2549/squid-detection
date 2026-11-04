from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Squid Detection AI", page_icon="🦑", layout="centered"
)

# หัวข้อเว็บไซต์
st.markdown(
    "<h1 style='text-align: center;'>🦑 ระบบตรวจจับปลาหมึกด้วย YOLO AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>อัปโหลดภาพปลาหมึกที่คุณต้องการตรวจสอบความแม่นยำของโมเดลผ่านหน้าเว็บได้ทันที</p>",
    unsafe_allow_html=True,
)

# โหลดโมเดล YOLO ที่เทรนเสร็จแล้ว (ใช้ไฟล์ best.pt ของปลาหมึก)
@st.cache_resource
def load_model():
  # โหลดไฟล์ best.pt ที่อยู่ในโฟลเดอร์เดียวกันกับ app.py
  model = YOLO("best.pt")
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f"ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบว่ามีไฟล์ 'best.pt' อยู่ในโปรเจกต์หรือไม่:"
      f" {e}"
  )

# ส่วนสำหรับอัปโหลดรูปภาพ
uploaded_file = st.file_uploader(
    "เลือกไฟล์รูปภาพ (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # เปิดภาพที่ผู้ใช้อัปโหลด
  image = Image.open(uploaded_file)

  # จัดlayout แสดงภาพต้นฉบับและผลลัพธ์
  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### รูปต้นฉบับ")
    st.image(image, use_column_width=True)

  with col2:
    st.markdown("### ผลลัพธ์การ Detection")

    with st.spinner("กำลังประมวลผลด้วยโมเดล YOLO AI..."):
      # แปลงภาพเป็น numpy array เพื่อให้ YOLO ประมวลผล
      img_array = np.array(image)

      # รันโมเดล Predict
      results = model.predict(img_array, conf=0.25)
      res = results[0]

      # ดึงภาพที่มีกรอบ bounding box ออกมา (แปลงกลับเป็น PIL Image)
      res_plotted = res.plot()  # ได้เป็น BGR numpy array
      res_image = Image.fromarray(
          res_plotted[..., ::-1]
      )  # แปลง BGR เป็น RGB

      # แสดงภาพผลลัพธ์
      st.image(res_image, use_column_width=True)

  st.success("✨ วิเคราะห์สำเร็จ!")

  # แสดงข้อมูลการตรวจจับเพิ่มเติม (ถ้ามี)
  boxes = res.boxes
  if len(boxes) > 0:
    st.markdown("### 📊 รายละเอียดการตรวจจับ:")
    for i, box in enumerate(boxes):
      class_id = int(box.cls[0])
      class_name = model.names[class_id]
      confidence = float(box.conf[0]) * 100
      st.info(
          f"🔹 ตรวจพบ: **{class_name}** (ความมั่นใจ:"
          f" {confidence:.1f}%)"
      )
  else:
    st.warning("ไม่พบวัตถุที่ตรงกับโมเดลในภาพนี้ ลองเปลี่ยนรูปภาพดูใหม่อีกครั้ง")