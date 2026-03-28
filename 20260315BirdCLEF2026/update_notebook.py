import json

with open('d:/kaggle/20260315BirdCLEF2026/notebook_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # Update train_model() code
    if 'def train_model():' in source:
        new_source = """def train_model():
    \"\"\"学習の実行関数\"\"\"
    
    # 準備：先ほどフィルタリングしたような綺麗なデータ(high_quality_train)を使用すると良いです。
    # カリキュラム学習の簡易版として、今回はレーティング3.0以上のデータで初期学習を行います。
    # 実装例: まず high_quality_train で学習し、のちに全体で学習するなど。
    train_used_df = high_quality_train if 'high_quality_train' in globals() else train_df
    
    # データローダーの作成
    train_dataset = BirdDataset(train_used_df, CONFIG["base_dir"], CONFIG)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    
    num_classes = train_dataset.num_classes
    
    # モデル、損失関数、最適化手法の定義
    model = BirdCLEFModel(CONFIG["model_name"], num_classes).to(device)
    
    # クラス不均衡対策の重み (species_weights) をLossに適用
    weights_tensor = torch.ones(num_classes)
    if 'species_weights' in globals():
        for row in species_weights.iter_rows(named=True):
            label = row['primary_label']
            if label in train_dataset.label_to_idx:
                idx = train_dataset.label_to_idx[label]
                # 重みが大きすぎると勾配爆発の原因になるため上限を設ける
                weights_tensor[idx] = min(row['weight'] * len(train_df), 10.0)
    
    weights_tensor = weights_tensor.to(device)
    
    # BirdCLEFは複数ラベルになり得るため、CrossEntropyではなくBCE(Binary Cross Entropy)を使います
    # ここで pos_weight として算出したクラスの重みを渡します
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor) 
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"Training started on {device}...")
    
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        running_loss = 0.0
        
        # プログレスバーの表示
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for images, targets in bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # フォワードパス
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # バックプロパゲーションとパラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f}")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), "efficientnet_b3_birdclef.pth")
    print("Model saved to efficientnet_b3_birdclef.pth")

# 実行（まずはエラーなく動くか少ないデータで試します）
train_model()
"""
        # Split by lines and keep newlines
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        # remove last empty newline
        if cell['source'][-1] == '\n':
            cell['source'].pop()
            
    # Update inference() code to add location mapping post-processing
    if 'def inference():' in source:
        new_source = """import glob
import math

def inference():
    \"\"\"テストデータ (soundscapes) に対する推論を行う\"\"\"
    # 1. 保存したモデルのロード
    model = BirdCLEFModel(CONFIG["model_name"], num_classes=len(train_df["primary_label"].unique()))
    
    try:
        model.load_state_dict(torch.load("efficientnet_b3_birdclef.pth", map_location=device))
        print("Trained model weights loaded.")
    except Exception as e:
        print(f"Warning: Could not load weights. Using untrained model for layout testing. ({e})")
        
    model.to(device)
    model.eval() # 推論モードに切り替え
    
    # 2. テスト音声の準備
    test_audio_dir = BASE_DIR + "/test_soundscapes"
    test_files = glob.glob(f"{test_audio_dir}/*.ogg")
    print(f"Found {len(test_files)} test files.")
    
    # 変換器
    mel_transform = T.MelSpectrogram(sample_rate=CONFIG["sample_rate"], n_mels=CONFIG["n_mels"], n_fft=1024, hop_length=512).to(device)
    amplitude_to_db = T.AmplitudeToDB().to(device)
    target_frames = CONFIG["sample_rate"] * CONFIG["duration"] 

    submission_dict = {"row_id": []}
    labels = train_df["primary_label"].unique().to_list()
    for label in labels:
        submission_dict[label] = []

    # ロケーションマッピングを辞書化して高速アクセス(推論後処理用)
    # location_map は (lat, lon) -> [possible species]
    loc_dict = {}
    if 'location_map' in globals():
        for row in location_map.iter_rows(named=True):
            lat, lon = row['latitude'], row['longitude']
            possible = row['possible_species']
            if possible is not None:
                loc_dict[(lat, lon)] = set(possible.to_list())

    # 3. 推論ループ
    with torch.no_grad():
        for filepath in tqdm(test_files, desc="Predicting Test Files"):
            filename = os.path.basename(filepath)
            filename_id = filename.split('.')[0]
            
            # 緯度・経度情報のダミー抽出(実際はtestのメタデータやファイル名から取得)
            # ファイル名形式: BC2026_Test_<ファイルID>_<場所>_<日にち>_<時間>.ogg
            # (ここでは実際の推論時にtest_soundscapes_labels.csv等が必要な場合がありますが、
            # サンプルのため全ての予測確率に微弱なペナルティを課す形式で実装します。
            # 実際の生息情報が特定できた場合は、以下のようにペナルティをかけます)
            
            # テストデータの場所が分かっていると仮定したロケーション・フィルタリング変数
            current_lat, current_lon = -8.0, -60.0 # ダミー座標
            grid_size = 5.0
            test_lat_grid = round(current_lat / grid_size) * grid_size
            test_lon_grid = round(current_lon / grid_size) * grid_size
            
            allowed_species = loc_dict.get((test_lat_grid, test_lon_grid), None)

            try:
                waveform, sr = torchaudio.load(filepath)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                continue
            
            num_chunks = math.ceil(waveform.shape[1] / target_frames)
            
            for i in range(num_chunks):
                start_frame = i * target_frames
                end_frame = min((i + 1) * target_frames, waveform.shape[1])
                chunk = waveform[:, start_frame:end_frame].to(device)
                
                if chunk.shape[1] < target_frames:
                    pad_size = target_frames - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
                mel_spec = mel_transform(chunk)
                mel_db = amplitude_to_db(mel_spec)
                mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
                
                image = mel_db.expand(3, -1, -1).unsqueeze(0)
                image = torch.nn.functional.interpolate(image, size=(CONFIG["image_size"], CONFIG["image_size"]), mode="bilinear")
                
                logits = model(image)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                
                # --- ロケーションフィルタリング (後処理) ---
                if allowed_species is not None:
                    for j, label in enumerate(labels):
                        # その地域に生息しない鳥の予測確率を 0.1 倍に下げる（ペナルティ）
                        if label not in allowed_species:
                            probs[j] *= 0.1 
                
                end_seconds = (i + 1) * 5
                row_id = f"{filename_id}_{end_seconds}"
                
                submission_dict["row_id"].append(row_id)
                for j, label in enumerate(labels):
                    submission_dict[label].append(probs[j])

    # 4. DataFrame に変換して保存
    submission_df = pl.DataFrame(submission_dict)
    submission_df.to_pandas().to_csv("submission.csv", index=False)
    print("Inference completed! Saved to submission.csv")
    return submission_df

# 実行
sub_df = inference()
display(sub_df.head())
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        if cell['source'][-1] == '\n':
            cell['source'].pop()

with open('d:/kaggle/20260315BirdCLEF2026/notebook_1_updated.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
