import fitz
import cv2
import numpy as np
import os
import sys

def extract_chemical_structures_from_pdf_v3(pdf_path, output_dir, dpi=300):
    """
    V3版本：颜色感知提取
    
    新功能:
    1.  [保留颜色] 最终输出为彩色图像。
    2.  [颜色识别] 使用HSV颜色空间来识别特定的前景颜色（如蓝色）。
    3.  [智能蒙版] 合并黑色和蓝色的识别结果，生成一个更干净的前景蒙版。
    """
    print(f"🚀 开始处理PDF (V3 - 颜色感知版): {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        print(f"\n📄 正在处理第 {page_num + 1}/{len(doc)} 页...")
        
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        # 1. 准备图像 (原始彩色图、灰度图、HSV图)
        img_cv_bgr = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n),
                                  cv2.COLOR_BGRA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2HSV)

        # 2. 创建颜色蒙版
        # 蒙版1: 识别黑色/深灰色物体 (使用灰度图)
        _, black_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 蒙版2: 识别蓝色物体 (使用HSV图)
        # 定义蓝色的HSV范围。这个范围可能需要微调。
        # H(色相): 100-140 是一个比较通用的蓝色范围
        # S(饱和度): 50-255 表示要有一定的色彩，不是灰色
        # V(亮度): 50-255 表示不能是纯黑色
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 合并两个蒙版：只要是黑色的或是蓝色的，都算作前景
        combined_mask = cv2.bitwise_or(black_mask, blue_mask)

        # 3. 后续处理 (与V2类似，但使用新的'combined_mask')
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        structures_on_page = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 100 or h < 100: continue

            # 填充率计算要在 *未膨胀* 的合并蒙版上进行，以保证准确性
            roi_mask = combined_mask[y:y+h, x:x+w]
            non_zero_pixels = cv2.countNonZero(roi_mask)
            total_pixels = w * h
            if total_pixels == 0: continue
            fill_ratio = non_zero_pixels / total_pixels
            
            # 使用您找到的完美参数！
            if not (0.001 < fill_ratio < 0.1):
                continue

            structures_on_page += 1
            total_structures_found = (locals().get('total_structures_found', 0) + 1)
            
            # 4. 在原始彩色图上裁剪！
            padding = 15
            cropped_image = img_cv_bgr[max(0, y-padding):min(img_cv_bgr.shape[0], y+h+padding), 
                                     max(0, x-padding):min(img_cv_bgr.shape[1], x+w+padding)]

            filename = f"page_{page_num+1}_structure_{structures_on_page}_color.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_image)

        if structures_on_page > 0:
            print(f"   ✅ 在本页找到 {structures_on_page} 个可能的彩色结构。")
        else:
            print("   ⚪ 在本页未找到符合条件的结构。")

    doc.close()
    # Safely retrieve and print total_structures_found
    total_found = locals().get('total_structures_found', 0)
    print(f"\n🎉 处理完成！总共提取了 {total_found} 个可能的化学结构。")
    print(f"请检查目录: {output_dir}")

# --- 主程序 ---
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ 当前目录下未找到PDF文件。")
        sys.exit(1)
    print("📂 找到以下PDF文件:")
    for i, pdf in enumerate(pdf_files):
        print(f"  {i+1}: {pdf}")
    try:
        choice = int(input(f"👉 请选择要处理的文件编号 (1-{len(pdf_files)}): ")) - 1
        selected_pdf = pdf_files[choice]
    except (ValueError, IndexError):
        print("❌ 无效选择。")
        sys.exit(1)
    output_folder = f"extracted_structures_{os.path.splitext(selected_pdf)[0]}_v3_color"
    extract_chemical_structures_from_pdf_v3(selected_pdf, output_folder)