import streamlit as st
#import cv2
from PIL import Image
import os
from ultralytics import YOLO

def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None

def process_image(image, model):
    try:
        results = model(image)
        result_image = results[0].plot()
        #result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
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

    if uploaded_file is not None:
        with st.spinner(text='资源加载中...'):
            st.sidebar.image(uploaded_file, caption="上传的图像", use_container_width =True)
            image= Image.open(uploaded_file)

            if st.sidebar.button("执行检测"):
                if model:
                    st.write("正在执行目标检测...")
                    result_image, results = process_image(image, model)

                    # 显示结果图像
                    st.image(result_image, caption="检测结果", use_column_width=True)

                    # 显示检测到的目标信息
                    if results is not None and len(results.boxes) > 0:
                        st.write("## 检测结果")

                        # 创建结果表格
                        result_data = []
                        for i, box in enumerate(results.boxes):
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            # 获取置信度
                            conf = box.conf[0].item()

                            # 获取类别
                            cls = int(box.cls[0].item())
                            cls_name = results.names[cls]

                            result_data.append({
                                "序号": i + 1,
                                "类别": cls_name,
                                "置信度": f"{conf:.2f}",
                                "位置": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                            })

                        # 显示结果表格
                        st.table(result_data)
                    else:
                        st.info("未检测到任何目标")

