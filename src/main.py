# process_pdfs.py (Version 5 - Production Final)

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

# --- 核心提取函数 (V5 - 增强了错误报告) ---
def extract_chemical_structures_from_pdf_v5(pdf_path: Path, output_dir: Path, dpi: int = 300):
    """
    V5版本：增强了对保存失败的调试信息。
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

            _, black_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            combined_mask = cv2.bitwise_or(black_mask, blue_mask)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            structures_on_page = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 100 or h < 100: continue
                roi_mask = combined_mask[y:y+h, x:x+w]
                if cv2.countNonZero(roi_mask) == 0: continue
                fill_ratio = cv2.countNonZero(roi_mask) / (w * h)
                if not (0.001 < fill_ratio < 0.09): continue

                try:
                    padding = 15
                    cropped_image = img_cv_bgr[max(0, y-padding):min(img_cv_bgr.shape[0], y+h+padding), 
                                             max(0, x-padding):min(img_cv_bgr.shape[1], x+w+padding)]
                    if cropped_image.size == 0: continue
                    
                    filename = f"page_{page_num+1}_structure_{structures_on_page + 1}_color.png"
                    filepath = output_dir / filename
                    
                    success = cv2.imwrite(str(filepath), cropped_image)

                    if success:
                        structures_on_page += 1
                        total_structures_found_in_file += 1
                    else:
                        # V5 改进点：提供更详细的失败诊断信息
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


# --- 主程序 (V5 - 使用哈希命名并创建清单) ---
def main():
    print("🚀 启动化学结构批量提取程序 (V5 - 生产级最终版)...")
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
    # 'w'模式表示每次运行都重新创建一个新的清单文件
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        manifest_file.write("Original_Filename,Hashed_Directory_Name\n")
        
        for pdf_path in pdf_files:
            # V5 改进点：使用MD5哈希值作为目录名
            original_filename = pdf_path.name
            # 使用utf-8编码文件名以处理特殊字符
            hashed_name = hashlib.md5(original_filename.encode('utf-8')).hexdigest()
            
            output_dir = RESULTS_DIR / hashed_name
            
            # 写入清单文件
            manifest_file.write(f'"{original_filename}",{hashed_name}\n')
            
            count = extract_chemical_structures_from_pdf_v5(pdf_path, output_dir)
            grand_total_structures += count
            print(f"   - 结果目录: {output_dir}")
            print("-" * 50)

    print("🎉🎉🎉 所有PDF处理完毕！ 🎉🎉🎉")
    print(f"总共成功保存了 {grand_total_structures} 个化学结构。")
    print(f"💡 文件名与目录对应关系已保存在: {manifest_path}")

if __name__ == "__main__":
    main()