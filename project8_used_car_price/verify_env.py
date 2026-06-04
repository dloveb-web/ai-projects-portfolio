#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境验证脚本"""

import sys
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

print("\n" + "="*60)
print("测试包导入...")
print("="*60)

packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'catboost': 'catboost',
    'lightgbm': 'lightgbm',
    'xgboost': 'xgboost',
    'torch': 'torch',
}

for module_name, package_name in packages.items():
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:15} - {version}")
    except ImportError as e:
        print(f"❌ {package_name:15} - 导入失败: {e}")

print("\n" + "="*60)
print("环境验证完成！")
print("="*60)
