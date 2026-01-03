# process_pdfs.py (Version 6 - Enhanced Structure Recognition)

import fitz
import cv2
import numpy as np
import sys
import hashlib
from pathlib import Path

# 1. 从您的 paths.py 文件中导入路径
try:
    from paths import PDFS_DIR, ROOT_DIR
except ImportError:
    print("错误：无法找到 paths.py 文件。请确保它和本脚本在同一个目录下。")
    sys.exit(1)

# --- 新增：高级结构验证函数 ---
def is_likely_chemical_structure(roi_mask, contour, original_image_roi, debug_mode=True):
    """
    使用多种特征判断一个区域是否可能是化学结构
    debug_mode=True 时会打印详细的检查信息
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    if debug_mode:
        print(f"      检查区域: {w}x{h}")
    
    # 1. 基本尺寸过滤 - 大幅放宽条件
    if w < 50 or h < 40:
        if debug_mode: print(f"      ❌ 尺寸太小: {w}x{h}")
        return False, "尺寸太小"
    
    # 2. 填充率检查 - 这是最重要的，先只用这个
    fill_ratio = cv2.countNonZero(roi_mask) / (w * h)
    if debug_mode: print(f"      填充率: {fill_ratio:.4f}")
    
    if not (0.001 < fill_ratio < 0.15):  # 进一步放宽
        if debug_mode: print(f"      ❌ 填充率异常: {fill_ratio:.4f}")
        return False, f"填充率异常: {fill_ratio:.4f}"
    
    # 3. 长宽比检查 - 放宽条件
    aspect_ratio = max(w, h) / min(w, h)
    if debug_mode: print(f"      长宽比: {aspect_ratio:.2f}")
    if aspect_ratio > 15:  # 大幅放宽
        if debug_mode: print(f"      ❌ 长宽比过大: {aspect_ratio:.2f}")
        return False, f"长宽比异常: {aspect_ratio:.2f}"
    
    # 暂时注释掉其他严格的检查，先看基本的能不能工作
    """
    # 4. 边缘密度检查
    edges = cv2.Canny(roi_mask, 50, 150)
    edge_density = cv2.countNonZero(edges) / (w * h)
    if debug_mode: print(f"      边缘密度: {edge_density:.4f}")
    if edge_density < 0.005:  # 放宽条件
        if debug_mode: print(f"      ❌ 边缘密度过低: {edge_density:.4f}")
        return False, f"边缘密度过低: {edge_density:.4f}"
    """
    
    # 简单的图表检测 - 只检测明显的网格
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//5, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//5))
    
    horizontal_lines = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, vertical_kernel)
    
    h_line_density = cv2.countNonZero(horizontal_lines) / (w * h)
    v_line_density = cv2.countNonZero(vertical_lines) / (w * h)
    
    if debug_mode: print(f"      线密度 H:{h_line_density:.4f}, V:{v_line_density:.4f}")
    
    # 只过滤明显的图表
    if h_line_density > 0.05 and v_line_density > 0.05:
        if debug_mode: print(f"      ❌ 疑似网格图表")
        return False, f"疑似图表 (H线密度: {h_line_density:.4f}, V线密度: {v_line_density:.4f})"
    
    if debug_mode: print(f"      ✅ 通过检查")
    return True, "通过所有检查"

def extract_chemical_structures_from_pdf_v6(pdf_path: Path, output_dir: Path, dpi: int = 300):
    """
    V6版本：增强的化学结构识别，减少误判
    """
    print(f"📄 正在处理: {pdf_path.name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"   ❌ 打开PDF文件失败: {e}")
        return 0

    total_structures_found_in_file = 0

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_cv_bgr = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n),
                                      cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2HSV)

            # 黑色检测 (稍微调整阈值)
            _, black_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            
            # 蓝色检测
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 红色检测 (新增，因为化学结构中常有红色标记)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # 组合所有颜色掩码
            combined_mask = cv2.bitwise_or(black_mask, cv2.bitwise_or(blue_mask, red_mask))
            
            # 形态学处理 (减少膨胀以保持精度)
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            structures_on_page = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 提取ROI掩码和原图像
                roi_mask = combined_mask[y:y+h, x:x+w]
                original_roi = img_cv_bgr[y:y+h, x:x+w]
                
                # 使用增强的验证函数 (开启调试模式)
                is_valid, reason = is_likely_chemical_structure(roi_mask, contour, original_roi, debug_mode=True)
                
                if not is_valid:
                    print(f"   - 过滤原因: {reason}")
                    continue

                try:
                    padding = 20  # 稍微增加padding
                    cropped_image = img_cv_bgr[max(0, y-padding):min(img_cv_bgr.shape[0], y+h+padding), 
                                             max(0, x-padding):min(img_cv_bgr.shape[1], x+w+padding)]
                    if cropped_image.size == 0: continue
                    
                    filename = f"page_{page_num+1}_structure_{structures_on_page + 1}_color.png"
                    filepath = output_dir / filename
                    
                    success = cv2.imwrite(str(filepath), cropped_image)

                    if success:
                        structures_on_page += 1
                        total_structures_found_in_file += 1
                        print(f"   ✓ 保存结构: {filename} (尺寸: {w}x{h})")
                    else:
                        print(f"   - 警告: OpenCV未能保存图片 {filename}。")
                        print(f"     - 目标路径: {filepath}")
                        print(f"     - 路径长度: {len(str(filepath))} (提示: Windows最大路径通常为260)")
                        print(f"     - 图像尺寸: {cropped_image.shape}")

                except Exception as e:
                    print(f"   - 错误: 保存 page {page_num+1} 的一个结构时发生异常: {e}")
        
        except Exception as page_e:
            print(f"   - 严重错误: 处理第 {page_num + 1} 页时失败: {page_e}")

    print(f"   ✅ 完成！在此文件中成功保存 {total_structures_found_in_file} 个结构。")
    doc.close()
    return total_structures_found_in_file


# --- 主程序 (V6) ---
def main():
    print("🚀 启动化学结构批量提取程序 (V6 - 增强识别精度版)...")
    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"📂 结果将保存在: {RESULTS_DIR}")
    
    # 清单文件，用于记录哈希与原文件名的对应关系
    manifest_path = RESULTS_DIR / "manifest.csv"
    
    pdf_files = sorted(list(PDFS_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"❌ 在目录 {PDFS_DIR} 中未找到任何PDF文件。")
        return

    print(f"🔎 找到 {len(pdf_files)} 个PDF文件准备处理。")
    print("-" * 50)

    grand_total_structures = 0
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        manifest_file.write("Original_Filename,Hashed_Directory_Name\n")
        
        for pdf_path in pdf_files:
            original_filename = pdf_path.name
            hashed_name = hashlib.md5(original_filename.encode('utf-8')).hexdigest()
            
            output_dir = RESULTS_DIR / hashed_name
            
            # 写入清单文件
            manifest_file.write(f'"{original_filename}",{hashed_name}\n')
            
            count = extract_chemical_structures_from_pdf_v6(pdf_path, output_dir)
            grand_total_structures += count
            print(f"   - 结果目录: {output_dir}")
            print("-" * 50)

    print("🎉🎉🎉 所有PDF处理完毕！ 🎉🎉🎉")
    print(f"总共成功保存了 {grand_total_structures} 个化学结构。")
    print(f"💡 文件名与目录对应关系已保存在: {manifest_path}")

if __name__ == "__main__":
    main()