import os
import sys
from PIL import Image
import subprocess

# 首先尝试导入，如果失败则安装
try:
    import fitz  # PyMuPDF
except ImportError:
    print("🔧 安装 PyMuPDF...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyMuPDF'])
    import fitz

def install_requirements():
    """安装必要的依赖"""
    print("🔧 检查并安装必要依赖...")
    
    try:
        import fitz
        print("✅ PyMuPDF 已安装")
    except ImportError:
        print("📦 安装 PyMuPDF...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyMuPDF'])
        import fitz
    
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image 已安装")
    except ImportError:
        print("📦 安装 pdf2image...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdf2image'])
        except:
            print("⚠️  pdf2image 安装失败，将使用其他方法")

def extract_images_with_pymupdf(pdf_path, output_dir):
    """
    使用 PyMuPDF 提取PDF中的图像
    """
    print(f"🔍 使用 PyMuPDF 分析PDF...")
    
    try:
        doc = fitz.open(pdf_path)
        image_count = 0
        
        print(f"📊 PDF信息:")
        print(f"  - 总页数: {len(doc)}")
        print(f"  - 文档元数据: {doc.metadata}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历每一页
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(f"\n📄 处理第 {page_num + 1} 页:")
            
            # 获取页面中的图像列表
            image_list = page.get_images(full=True)
            print(f"  🖼️  找到 {len(image_list)} 个图像对象")
            
            # 提取每个图像
            for img_index, img in enumerate(image_list):
                try:
                    # 获取图像引用
                    xref = img[0]
                    
                    # 提取图像数据
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 保存图像
                    image_count += 1
                    filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # 获取图像信息
                    try:
                        pil_img = Image.open(filepath)
                        width, height = pil_img.size
                        print(f"    ✅ {filename} - 尺寸: {width}x{height}")
                        
                        # 如果图像足够大，可能包含化学结构
                        if width > 100 and height > 100:
                            print(f"       🧪 可能包含化学结构 (尺寸合适)")
                        
                    except Exception as e:
                        print(f"    ⚠️  无法分析图像: {e}")
                
                except Exception as e:
                    print(f"    ❌ 提取图像 {img_index + 1} 失败: {e}")
            
            # 如果页面没有图像，尝试将整页转换为图像
            if not image_list:
                try:
                    # 将页面渲染为图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2倍缩放
                    pix = page.get_pixmap(matrix=mat)
                    
                    page_filename = f"page_{page_num + 1}_full.png"
                    page_filepath = os.path.join(output_dir, page_filename)
                    pix.save(page_filepath)
                    
                    print(f"    ✅ 保存整页: {page_filename}")
                    image_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 保存整页失败: {e}")
        
        doc.close()
        print(f"\n🎉 PyMuPDF 提取完成！总共提取了 {image_count} 个图像")
        return image_count > 0
        
    except Exception as e:
        print(f"❌ PyMuPDF 提取失败: {e}")
        return False

def extract_text_with_pymupdf(pdf_path, output_dir):
    """
    使用 PyMuPDF 提取PDF文本并分析化学内容
    """
    print(f"\n📝 提取PDF文本内容...")
    
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        chemical_keywords = [
            'molecule', 'compound', 'synthesis', 'reaction', 'chemical', 
            'organic', 'anion', 'cation', 'ion', 'acid', 'base',
            'benzene', 'phenyl', 'methyl', 'ethyl', 'hydroxyl',
            'carboxyl', 'amino', 'nitro', 'halogen'
        ]
        
        chemical_sentences = []
        
        # 提取每页文本
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            all_text += f"\n--- 第 {page_num + 1} 页 ---\n{page_text}"
            
            # 查找化学相关句子
            sentences = page_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # 过滤太短的句子
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in chemical_keywords):
                        chemical_sentences.append(f"页{page_num + 1}: {sentence}")
        
        doc.close()
        
        # 保存完整文本
        text_file = os.path.join(output_dir, "full_text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(all_text)
        print(f"  ✅ 完整文本保存至: {text_file}")
        
        # 保存化学相关句子
        if chemical_sentences:
            chem_file = os.path.join(output_dir, "chemical_sentences.txt")
            with open(chem_file, 'w', encoding='utf-8') as f:
                f.write("=== 化学相关句子 ===\n\n")
                for i, sentence in enumerate(chemical_sentences[:50], 1):  # 限制前50个
                    f.write(f"{i}. {sentence}\n\n")
            
            print(f"  ✅ 化学句子保存至: {chem_file} ({len(chemical_sentences)} 个)")
        
        return len(chemical_sentences)
        
    except Exception as e:
        print(f"❌ 文本提取失败: {e}")
        return 0

def safe_chemdataextractor_analysis(pdf_path, output_dir):
    """
    安全地使用 ChemDataExtractor 进行分析
    """
    print(f"\n🧪 尝试 ChemDataExtractor 分析...")
    
    try:
        from chemdataextractor.doc import Document
        
        # 设置递归限制，避免无限递归
        import sys
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000)  # 降低递归限制
        
        try:
            # 只读取前几页避免复杂性
            print("  📖 读取PDF文档（限制复杂度）...")
            
            # 先用PyMuPDF读取文本，然后用CDE分析
            doc_fitz = fitz.open(pdf_path)
            sample_text = ""
            
            # 只取前3页的文本
            for page_num in range(min(3, len(doc_fitz))):
                page = doc_fitz.load_page(page_num)
                sample_text += page.get_text() + "\n"
            
            doc_fitz.close()
            
            if sample_text.strip():
                print("  🔍 使用文本创建CDE文档...")
                cde_doc = Document(sample_text[:10000])  # 限制文本长度
                
                # 尝试提取化学实体
                try:
                    cems = cde_doc.cems
                    if cems:
                        chem_entities_file = os.path.join(output_dir, "cde_chemical_entities.txt")
                        with open(chem_entities_file, 'w', encoding='utf-8') as f:
                            f.write("=== ChemDataExtractor 化学实体 ===\n\n")
                            for i, cem in enumerate(cems[:20], 1):  # 限制数量
                                f.write(f"{i}. {cem.text}\n")
                        
                        print(f"  ✅ CDE化学实体: {len(cems)} 个 -> {chem_entities_file}")
                    else:
                        print("  ⚠️  CDE未找到化学实体")
                
                except Exception as e:
                    print(f"  ⚠️  CDE化学实体提取失败: {e}")
            
        finally:
            # 恢复原始递归限制
            sys.setrecursionlimit(original_limit)
        
        return True
        
    except Exception as e:
        print(f"  ❌ CDE分析失败: {e}")
        return False

def analyze_chemical_structures(output_dir):
    """
    分析提取的图像，寻找可能的化学结构
    """
    print(f"\n🔬 分析提取的图像...")
    
    try:
        image_files = [f for f in os.listdir(output_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("  ⚠️  没有图像文件可供分析")
            return
        
        print(f"  📊 分析 {len(image_files)} 个图像文件:")
        
        structure_candidates = []
        
        for img_file in image_files:
            img_path = os.path.join(output_dir, img_file)
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # 简单的启发式规则判断是否可能包含化学结构
                    is_candidate = False
                    reason = []
                    
                    # 尺寸合适
                    if 150 <= width <= 800 and 150 <= height <= 600:
                        is_candidate = True
                        reason.append("尺寸合适")
                    
                    # 分析图像特征
                    if img.mode == 'RGB':
                        # 转换为灰度分析
                        gray = img.convert('L')
                        
                        # 简单的方差分析（结构复杂度）
                        import numpy as np
                        img_array = np.array(gray)
                        variance = np.var(img_array)
                        
                        if variance > 1000:  # 有足够的变化，可能包含结构
                            is_candidate = True
                            reason.append("结构复杂")
                    
                    status = "✅ 候选" if is_candidate else "⚪ 普通"
                    reason_str = ", ".join(reason) if reason else "基础图像"
                    
                    print(f"    {status} {img_file} ({width}x{height}) - {reason_str}")
                    
                    if is_candidate:
                        structure_candidates.append(img_file)
            
            except Exception as e:
                print(f"    ❌ 分析 {img_file} 失败: {e}")
        
        # 保存分析结果
        if structure_candidates:
            analysis_file = os.path.join(output_dir, "structure_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("=== 化学结构候选图像 ===\n\n")
                for candidate in structure_candidates:
                    f.write(f"• {candidate}\n")
                f.write(f"\n总计: {len(structure_candidates)} 个候选图像")
            
            print(f"  🎯 找到 {len(structure_candidates)} 个结构候选 -> {analysis_file}")
        else:
            print(f"  ⚠️  未找到明显的化学结构候选")
        
    except Exception as e:
        print(f"❌ 图像分析失败: {e}")

def main():
    """主函数"""
    print("🚀 稳定版PDF化学结构提取器")
    print("=" * 50)
    
    # 安装依赖
    install_requirements()
    
    # 寻找PDF文件
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ 未找到PDF文件")
        return
    
    print(f"\n📂 找到PDF文件:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file}")
    
    # 选择文件
    if len(pdf_files) == 1:
        selected_pdf = pdf_files[0]
        print(f"\n🎯 处理: {selected_pdf}")
    else:
        try:
            choice = int(input(f"选择文件 (1-{len(pdf_files)}): ")) - 1
            selected_pdf = pdf_files[choice]
        except (ValueError, IndexError):
            print("❌ 无效选择")
            return
    
    # 创建输出目录
    base_name = os.path.splitext(selected_pdf)[0]
    output_folder = f"stable_extraction_{base_name}"
    
    print(f"\n🎬 开始处理...")
    print(f"📁 输出目录: {output_folder}")
    
    # 多步骤处理
    steps_completed = 0
    
    # 步骤1: PyMuPDF图像提取
    if extract_images_with_pymupdf(selected_pdf, output_folder):
        steps_completed += 1
    
    # 步骤2: 文本提取和分析
    if extract_text_with_pymupdf(selected_pdf, output_folder):
        steps_completed += 1
    
    # 步骤3: ChemDataExtractor分析（可选）
    if safe_chemdataextractor_analysis(selected_pdf, output_folder):
        steps_completed += 1
    
    # 步骤4: 图像结构分析
    analyze_chemical_structures(output_folder)
    steps_completed += 1
    
    # 结果总结
    print(f"\n🎉 处理完成！")
    print(f"✅ 完成步骤: {steps_completed}/4")
    print(f"📁 结果位置: {output_folder}")
    
    if os.path.exists(output_folder):
        files = os.listdir(output_folder)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        texts = [f for f in files if f.lower().endswith('.txt')]
        
        print(f"\n📊 提取统计:")
        print(f"  🖼️  图像: {len(images)} 个")
        print(f"  📄 文本: {len(texts)} 个")
        
        print(f"\n💡 建议下一步:")
        print(f"  1. 查看 {output_folder} 目录")
        print(f"  2. 检查候选化学结构图像")
        print(f"  3. 阅读化学相关文本片段")

if __name__ == "__main__":
    main()