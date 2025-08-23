import fitz  # PyMuPDF
import os
import glob
import shutil
from PIL import Image
from rdkit import Chem

# --- 核心依赖：DECIMER ---
from DECIMER import predict_SMILES

# ==============================================================================
#  ✨ 新增：严格过滤的配置参数 ✨
# 您可以调整这个值。如果想更严格，就调高；如果想宽松些，就调低。
# 6是一个很好的起点，因为它至少需要一个像苯环这样的基本结构。
MINIMUM_HEAVY_ATOMS = 6
# ==============================================================================


# ==============================================================================
#  核心分析函数：现在包含严格的质量检查
# ==============================================================================
def is_strict_molecule_image(image_path):
    """
    使用 DECIMER 分析图片，并严格验证其是否为一个足够复杂的化学结构。
    
    :param image_path: 图片文件的路径。
    :return: (bool) 如果识别出复杂分子则返回True，否则返回False。
    """
    try:
        predicted_smiles = predict_SMILES(image_path).strip()
        print(f"  [DECIMER] Predicted SMILES: '{predicted_smiles}'")

        if not predicted_smiles:
            print("  [诊断] 失败原因: DECIMER 未能识别出任何SMILES字符串。")
            return False

        mol = Chem.MolFromSmiles(predicted_smiles)
        
        if mol is None:
            print(f"  [诊断] 失败原因: RDKit无法解析该SMILES '{predicted_smiles}'。")
            return False
        
        # --- ✨ 核心的严格过滤逻辑 ✨ ---
        # 我们计算分子中非氢原子的数量 (Heavy Atoms)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        print(f"  [诊断] 分子复杂度: {num_heavy_atoms} 个非氢原子。")

        # 只有当原子数量达到我们设定的最小阈值时，才认为是有效分子
        if num_heavy_atoms >= MINIMUM_HEAVY_ATOMS:
            print(f"  [诊断] 成功: 分子足够复杂，通过严格过滤器。")
            return True
        else:
            print(f"  [诊断] 失败原因: 分子过于简单 (少于 {MINIMUM_HEAVY_ATOMS} 个原子)，很可能是图表或噪点导致的假阳性。")
            return False

    except Exception as e:
        print(f"  [错误] 分析图片 {os.path.basename(image_path)} 时出错: {e}")
        return False

# ==============================================================================
#  主流程函数 (更新了调用的函数名)
# ==============================================================================
def process_pdfs_with_molecule_filter(pdf_folder, temp_folder, final_folder):
    """
    主函数：从PDF提取所有图片，然后对图片进行筛选。
    """
    # ... (这部分文件操作代码和之前完全一样，无需修改) ...
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)

    print("\n--- 第一步：从所有PDF中提取图片到临时文件夹 ---")
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"在文件夹 '{pdf_folder}' 中没有找到PDF文件。请检查路径。")
        return

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            pdf_filename = os.path.basename(pdf_path)
            print(f"正在处理文件: {pdf_filename}")
            for page_num in range(len(doc)):
                for img_index, img in enumerate(doc.get_page_images(page_num, full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"{os.path.splitext(pdf_filename)[0]}_p{page_num+1}_i{img_index+1}.{image_ext}"
                    temp_image_path = os.path.join(temp_folder, image_filename)
                    with open(temp_image_path, "wb") as f:
                        f.write(image_bytes)
            doc.close()
        except Exception as e:
            print(f"处理PDF文件 {pdf_path} 时出错: {e}")
    
    print("\n--- 第二步：分析临时文件夹中的每张图片并进行严格筛选 ---")
    all_temp_images = glob.glob(os.path.join(temp_folder, "*"))
    for image_path in all_temp_images:
        print(f"分析图片: {os.path.basename(image_path)}")
        # --- 调用我们新的、严格的过滤函数 ---
        if is_strict_molecule_image(image_path):
            shutil.move(image_path, os.path.join(final_folder, os.path.basename(image_path)))
            print(f"  -> [保留图片]")
        else:
            os.remove(image_path)
            print(f"  -> [丢弃图片]")

    if os.path.exists(temp_folder) and not os.listdir(temp_folder):
        os.rmdir(temp_folder)

    print("\n处理完成！最终筛选出的图片已保存到:", final_folder)

# ==============================================================================
#  程序主入口 (保持不变)
# ==============================================================================
if __name__ == "__main__":
    pdf_directory = R"D:\workplace\code\Python\VsCode\pictureFind\data\8.17(1)"
    temp_directory = "temp_images"
    final_directory = "final_strict_molecule_images" # 文件夹名也更新一下
    
    process_pdfs_with_molecule_filter(pdf_directory, temp_directory, final_directory)