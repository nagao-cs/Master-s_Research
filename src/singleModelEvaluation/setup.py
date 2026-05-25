"""
Single Model Evaluation - セットアップスクリプト
必要なディレクトリ構造を初期化
"""

from pathlib import Path
import os


def setup_evaluation_directory():
    """評価用ディレクトリ構造を作成"""
    
    base_dir = Path(__file__).parent
    
    # 必要なディレクトリ
    directories = [
        "results",
        "results/Town02/figures",
        "results/Town03/figures",
        "results/Town05/figures",
    ]
    
    print("Setting up directory structure...")
    print("-" * 80)
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ {full_path}")
    
    # README.md を作成
    readme_path = base_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Single Model Evaluation"""

