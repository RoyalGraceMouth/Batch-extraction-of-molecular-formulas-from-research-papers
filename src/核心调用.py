# 核心调用.py (最终方案 v4: 明确指定编码器和解码器)

import os
import glob
import shutil
from rdkit import Chem, RDLogger
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import torch

# --- 1. 导入 Hugging Face 库 ---
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, RobertaTokenizer, AutoProcessor

# 禁用 RDKit 的冗余日志
RDLogger.DisableLog('rdApp.*')

# --- 2. 配置参数 ---
MINIMUM_HEAVY_ATOMS = 6
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_INPUT_DIR = DATA_DIR / "pdfs"
FINAL_IMAGES_DIR = ROOT_DIR / "final_strict_molecule_images"

# --- 关键：分别指定编码器和解码器的模型名称 ---
ENCODER_MODEL = "google/vit-base-patch16-224-in21k"
DECODER_MODEL = "tonyai-vn/img2smiDec"

# --- 3. 模型加载与预测器 ---
class OcrPredictor:
    """
    通过手动组合编码器和解码器来加载模型并执行预测
    """
    def __init__(self, encoder_name: str, decoder_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载编码器: {encoder_name}")
        print(f"正在加载解码器: {decoder_name}")
        print(f"预测将使用设备: {self.device}")
        
        # --- 核心修正：使用 from_encoder_decoder_pretrained 进行手动组装 ---
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=encoder_name,
            decoder_pretrained_model_name_or_path=decoder_name,
        ).to(self.device)

        # 处理器也需要正确加载
        # AutoProcessor通常能处理好，它会找到对应的图像处理器和文本分词器
        self.processor = AutoProcessor.from_pretrained(decoder_name)
        
        print("模型组装加载成功！")

    def predict_smiles(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, max_length=512)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            print(f" [错误] 预测图片 {Path(image_path).name} 时失败: {e}")
            return ""

# ==============================================================================
#  (核心分析函数 和 主流程函数 无需任何修改)
# ==============================================================================
def is_strict_molecule_image(image_path: str, predictor: OcrPredictor) -> bool:
    predicted_smiles = predictor.predict_smiles(image_path)
    print(f"  [模型预测] SMILES: '{predicted_smiles}'")
    if not predicted_smiles:
        print("  [诊断] 失败: 模型未能识别。")
        return False
    mol = Chem.MolFromSmiles(predicted_smiles)
    if mol is None:
        print(f"  [诊断] 失败: RDKit无法解析。")
        return False
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    print(f"  [诊断] 分子复杂度: {num_heavy_atoms} 个非氢原子。")
    if num_heavy_atoms >= MINIMUM_HEAVY_ATOMS:
        print(f"  [诊断] 成功: 分子通过过滤器。")
        return True
    else:
        print(f"  [诊断] 失败: 分子过于简单。")
        return False

def main_process(pdf_folder, final_folder, predictor):
    temp_folder = ROOT_DIR / "temp_extracted_images"
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)
    print("\n--- 第一步：从PDF提取图片 ---")
    pdf_files = glob.glob(str(pdf_folder / "*.pdf"))
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            print(f"处理文件: {os.path.basename(pdf_path)}")
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref, _, w, h, _, _, _, name, _, _ = img
                    if min(w,h) < 50: continue
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_filename = f"{Path(pdf_path).stem}_p{page_num+1}_i{img_index+1}.{base_image['ext']}"
                    temp_image_path = temp_folder / image_filename
                    with open(temp_image_path, "wb") as f: f.write(image_bytes)
        except Exception as e: print(f"处理PDF {pdf_path} 时出错: {e}")
    print("\n--- 第二步：使用加载的模型进行筛选 ---")
    all_temp_images = glob.glob(str(temp_folder / "*.*"))
    for image_path in all_temp_images:
        print(f"分析图片: {os.path.basename(image_path)}")
        if is_strict_molecule_image(image_path, predictor):
            shutil.move(image_path, final_folder / os.path.basename(image_path))
            print(f"  -> [保留图片]")
        else:
            os.remove(image_path)
            print(f"  -> [丢弃图片]")
    if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
    print(f"\n处理完成！最终图片已保存到: {final_folder}")

if __name__ == "__main__":
    predictor = OcrPredictor(encoder_name=ENCODER_MODEL, decoder_name=DECODER_MODEL)
    main_process(
        pdf_folder=PDF_INPUT_DIR, 
        final_folder=FINAL_IMAGES_DIR, 
        predictor=predictor
    )