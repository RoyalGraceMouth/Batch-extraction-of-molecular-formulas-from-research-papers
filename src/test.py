#!/usr/bin/env python3
"""
简单的DECIMER测试脚本
使用您指定的图片路径进行测试
"""

from DECIMER import predict_SMILES
from rdkit import Chem

def test_image(image_path):
    """测试指定路径的图片"""
    print(f"🧪 DECIMER化学结构识别测试")
    print("=" * 50)
    print(f"📁 图片路径: {image_path}")
    
    try:
        # 使用DECIMER预测SMILES
        print(f"🔍 正在分析图片...")
        predicted_smiles = predict_SMILES(image_path)
        predicted_smiles = predicted_smiles.strip()
        
        print(f"📝 预测的SMILES: {predicted_smiles}")
        
        if not predicted_smiles:
            print("❌ 未生成SMILES字符串")
            return False
        
        # 验证SMILES有效性
        mol = Chem.MolFromSmiles(predicted_smiles)
        if mol is None:
            print("❌ 生成的SMILES无效，无法被RDKit解析")
            return False
        else:
            print("✅ SMILES字符串有效")
        
        # 检查芳香环
        has_aromatic = False
        aromatic_atoms = []
        
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                has_aromatic = True
                aromatic_atoms.append(f"{atom.GetSymbol()}{atom.GetIdx()}")
        
        print(f"🔍 包含芳香环: {has_aromatic}")
        
        if has_aromatic:
            print(f"🎉 成功！检测到芳香环")
            print(f"💍 芳香原子: {', '.join(aromatic_atoms)}")
            print(f"🎯 结果: 该图片将被保留")
        else:
            print(f"❌ 未检测到芳香环")
            print(f"🎯 结果: 该图片将被丢弃")
        
        # 显示分子详细信息
        print(f"\n📊 分子统计信息:")
        print(f"  • 原子总数: {mol.GetNumAtoms()}")
        print(f"  • 键总数: {mol.GetNumBonds()}")
        print(f"  • 环总数: {mol.GetRingInfo().NumRings()}")
        
        return has_aromatic
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 直接指定您的图片路径
    image_path = R"D:\workplace\code\Python\VsCode\pictureFind\data\d5cp00233h_p2_i2.png"  # 例如: r"D:\images\molecule.png"
    test_image(image_path)