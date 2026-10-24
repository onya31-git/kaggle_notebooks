# kaggle用のRepository

## ファイル命名規則
**[カテゴリ+実験番号]\_[説明]\_[追加情報].[拡張子]**
- **expeliment(実験用ノートブック)**:
Ex.)
    - exp001_baseline_lgbm.ipynb
    - exp002_add_leg_features.ipynb
    - exp003_add_lag_features.py

- **model(学習済みモデル)**:
Ex.)
    - mdl_exp001_fold0.pkl
    - mdl_exp001_fold1.pkl
    - mdl_exp002_fulltrain.pth

- **submit(提出用ファイル)**:
Ex.)
    - sub_exp001_cv0.854.csv
    - sub_exp002_cv0.861.csv
    - sub_ensemble_exp001_exp002.csv

- **others(その他)**:
    - **data/** : 生データ用フォルダ
    - **feat/** : 特徴量用フォルダ
    - **cfg/**  : 設定ファイル(YAML等)
    - **log/**  : 実行結果フォルダ

