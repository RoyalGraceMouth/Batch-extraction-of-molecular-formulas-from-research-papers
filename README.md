## 项目简介

本项目旨在从PDF文件中提取化学结构图像，提供了一套高效的处理流程和增强的识别算法。通过结合OpenCV和PyMuPDF等工具，能够对PDF中的化学结构进行精准识别和裁剪。

## 环境配置

请确保已安装以下依赖项。推荐使用`conda`创建虚拟环境：

### 创建环境

1. 创建并激活虚拟环境：
    ```bash
    conda env create -f config/environment.yml
    conda activate chemo_extractor_env
    ```

2. 环境文件`environment.yml`的核心依赖包括：
    - **Python 3.9**: 提供兼容性和稳定性。
    - **ChemDataExtractor**: 用于化学数据的提取。
    - **Pillow**: 图像处理库。
    - **Jupyter**: 交互式开发环境。
    - **Matplotlib**: 数据可视化工具。
    - **tqdm**: 进度条显示工具。

3. 通过`pip`安装的关键包：
    - **rdkit-pypi**: 化学信息学工具包。
    - **DECIMER**: 化学结构识别工具。
    - **transformers**: 深度学习模型工具包。
    - **datasets**: 数据集管理工具。
    - **accelerate**: 加速深度学习训练。
    - **其他工具**: 如`rich`和`pyyaml`。

## 文件结构

项目的主要目录和文件说明如下：

```
pictureFind/
├── config/
│   └── environment.yml  # 环境配置文件
├── data/                # 数据目录（已被.gitignore忽略）
│   └── pdfs/            # 存放待处理的PDF文件
├── models/              # 模型目录（已被.gitignore忽略）
├── results/             # 结果输出目录
├── src/
│   ├── main.py          # 主程序，包含PDF处理逻辑
│   └── paths.py         # 路径配置文件
├── .gitignore           # Git忽略规则
├── LICENSE              # 项目许可证
└── README.md            # 项目说明文件
```

## 使用说明

### 运行主程序

1. 将待处理的PDF文件放入`data/pdfs/`目录。
2. 运行以下命令启动程序：
    ```bash
    python src/main.py
    ```
3. 程序会自动处理PDF文件，并将提取的化学结构图像保存到`results/`目录。

### 输出结果

- 提取的化学结构图像会按PDF文件名的哈希值存储在对应的子目录中。
- 程序会生成一个`manifest.csv`文件，记录原始文件名与哈希目录的对应关系。

## 注意事项

- 确保`data/pdfs/`目录中存在PDF文件，否则程序将无法运行。
- 如果路径过长导致保存失败，请检查操作系统的路径长度限制。
- 运行程序时，请确保安装的依赖项与`environment.yml`文件一致。

## 许可证

此项目基于 [MIT 许可证](LICENSE) 进行许可，允许用户自由使用、复制、修改和分发代码，但需保留原始许可证声明和版权信息。

## 联系方式

如有问题或反馈，请联系 34221766@qq.com
