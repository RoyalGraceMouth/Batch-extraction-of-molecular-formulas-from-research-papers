import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import sys

def extract_chemical_structures_from_pdf_v2(pdf_path, output_dir, dpi=300):
    """
    V2版本：通过形态学操作和特征分析提高准确性
    
    新功能:
    1.  [解决分割问题] 使用形态学“膨胀”操作来连接分子结构中断开的部分。
    2.  [解决误报问题] 计算轮廓的“填充率”，过滤掉logo、实心块等非分子式图像。
    """
    print(f"🚀 开始处理PDF (V2): {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ 打开PDF失败: {e}")
        return

    total_structures_found = 0

    for page_num in range(len(doc)):
        print(f"\n📄 正在处理第 {page_num + 1}/{len(doc)} 页...")
        
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_cv = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 反向二值化，使线条和文字为白色(255)，背景为黑色(0)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

        # --- V2 改进点 1: 形态学膨胀 ---
        # 创建一个小的矩形“核”来进行膨胀操作
        # (5,5)表示核的大小，可以调整。值越大，连接效果越强。
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # iterations=2 表示执行两次膨胀，增强连接效果
        dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # 在膨胀后的图像上寻找轮廓，这样原本分离的部分就会被一起找到
        contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        structures_on_page = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # --- 基本的几何过滤 ---
            min_width, min_height = 100, 100 # 适当提高最小尺寸，过滤掉小的噪声
            if w < min_width or h < min_height:
                continue

            # --- V2 改进点 2: 填充率过滤 ---
            # 我们需要在 *原始* 二值图像(thresh)上计算填充率，而不是膨胀后的
            roi = thresh[y:y+h, x:x+w]
            
            # 计算区域内非零像素（白色像素）的数量
            non_zero_pixels = cv2.countNonZero(roi)
            total_pixels = w * h
            
            # 避免除以零
            if total_pixels == 0:
                continue
                
            fill_ratio = non_zero_pixels / total_pixels
            
            # 设定一个合理的填充率阈值
            # 这个范围可以根据你的文档进行微调
            # 分子式通常不会太稀疏也不会太密集
            fill_ratio_min = 0.001  
            fill_ratio_max = 0.1

            if not (fill_ratio_min < fill_ratio < fill_ratio_max):
                # print(f"   - 过滤掉一个对象，填充率: {fill_ratio:.2f}") # (取消注释以进行调试)
                continue

            # 如果通过了所有过滤，我们就认为它是一个有效的化学结构
            structures_on_page += 1
            total_structures_found += 1
            
            padding = 15
            cropped_image = img_cv[max(0, y-padding):min(img_cv.shape[0], y+h+padding), 
                                 max(0, x-padding):min(img_cv.shape[1], x+w+padding)]

            filename = f"page_{page_num+1}_structure_{structures_on_page}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_image)

        if structures_on_page > 0:
            print(f"   ✅ 在本页找到 {structures_on_page} 个可能的结构。")
        else:
            print("   ⚪ 在本页未找到符合条件的结构。")

    doc.close()
    print(f"\n🎉 处理完成！总共提取了 {total_structures_found} 个可能的化学结构。")
    print(f"请检查目录: {output_dir}")


# --- 主程序 ---
if __name__ == "__main__":
    # (主程序代码与之前相同，此处省略)
    # ...
    # 确保调用的是 v2 版本的函数
    # extract_chemical_structures_from_pdf_v2(selected_pdf, output_folder)
    # ...
    # 为了方便您直接运行，我将主程序也附上
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
    output_folder = f"extracted_structures_{os.path.splitext(selected_pdf)[0]}_v2"
    extract_chemical_structures_from_pdf_v2(selected_pdf, output_folder)