import streamlit as st
import cv2
from PIL import Image
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd
import io

def login():
    st.sidebar.header("用户登录")
    username = st.sidebar.text_input("用户名")
    password = st.sidebar.text_input("密码", type="password")
    login_btn = st.sidebar.button("登录")
    if login_btn:
        if username == "admin" and password == "123456":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.sidebar.error("用户名或密码错误")


def preprocess_image(image):
    # 将PIL图像转为OpenCV格式（BGR）
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 调整为640x640
    image = cv2.resize(image, (640, 640))
    # CLAHE增强
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image

def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None

def process_image(image, model):
    try:
        results = model(image)
        result_image = results[0].plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image, results[0]
    except Exception as e:
        st.error(f"处理图像时出错: {e}")
        return None, None

if __name__ == '__main__':
    # 设置页面标题和布局
    st.set_page_config(
        page_title="Mass Detection and BI-RADS Classification",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
        st.stop()  # 阻止后续页面渲染

    # 主要功能区域
    st.title("乳腺图像肿块检测与BI-RADS分级")
    st.header('导言')
    st.text('这是一个用于Mammography图像中肿块识别与BI-RADS分级的系统，'
            '您可以使用它来上传您的胸部Mammography图像并检测您的肿块。')

    #模型加载
    model = load_model()

    # 侧边栏
    # 上传文件
    st.sidebar.header('乳腺图像输入')
    uploaded_file = st.sidebar.file_uploader(
        "请选择检测图片", type=['png', 'jpeg', 'jpg'])

    # 初始化 session state
    if "result_image" not in st.session_state:
        st.session_state["result_image"] = None
    if "result_data" not in st.session_state:
        st.session_state["result_data"] = None
    if "img_bytes" not in st.session_state:
        st.session_state["img_bytes"] = None

    if uploaded_file is not None:
        st.sidebar.image(uploaded_file, caption="上传的图像", use_container_width=True)
        image = Image.open(uploaded_file)

        if st.sidebar.button("执行检测"):
            if model:
                preprocessed_image = preprocess_image(image)
                result_image, results = process_image(preprocessed_image, model)
                if result_image is not None:
                    st.session_state["result_image"] = result_image
                    if results is not None and len(results.boxes) > 0:
                        result_data = []
                        for i, box in enumerate(results.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            cls_name = results.names[cls]
                            result_data.append({
                                "序号": i + 1,
                                "类别": cls_name,
                                "置信度": f"{conf:.2f}",
                                "位置": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                            })
                        df = pd.DataFrame(result_data)
                        st.session_state["result_data"] = df
                    else:
                        st.session_state["result_data"] = None

                    # 保存检测结果为PNG字节流
                    img_pil = Image.fromarray(result_image)
                    img_bytes = io.BytesIO()
                    img_pil.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    st.session_state["img_bytes"] = img_bytes
                else:
                    st.session_state["result_image"] = None
                    st.session_state["result_data"] = None
                    st.session_state["img_bytes"] = None

        # 展示检测结果（只要 session state 有就显示）
        if st.session_state["result_image"] is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(st.session_state["result_image"], caption="检测结果图像", use_container_width=True)

            if st.session_state["result_data"] is not None:
                st.write("## 检测结果")
                st.dataframe(st.session_state["result_data"], hide_index=True)
                st.download_button(
                    label="下载检测结果图像",
                    data=st.session_state["img_bytes"],
                    file_name="result.png",
                    mime="image/png"
                )
            else:
                st.info("未检测到任何目标")
