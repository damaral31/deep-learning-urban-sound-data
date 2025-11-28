"""
Script para detectar data leakage entre train, val e test
"""

import json
import os
from pathlib import Path
import config

def check_data_leakage(dataset_path, folds):
    """
    Verifica se há overlap de dados entre train/val/test
    """
    
    dataset_path = Path(dataset_path) / "urbansound8K"
    
    print("Analisando data leakage...\n")
    
    # Verificar cache se existir
    cache_path = config.PROJECT_ROOT / "datasets" / "augmentation_cache"
    cache_metadata_path = cache_path / "augmentation_cached_files_index.json"
    
    if os.path.exists(cache_metadata_path):
        print("=" * 60)
        print("VERIFICANDO CACHE")
        print("=" * 60)
        
        with open(cache_metadata_path) as f:
            cache_metadata = json.load(f)
        
        print(f"Total de arquivos em cache: {len(cache_metadata)}\n")
        
        # Agrupar por fold
        cache_by_fold = {}
        for item in cache_metadata:
            fold = item.get('fold')
            if fold not in cache_by_fold:
                cache_by_fold[fold] = []
            cache_by_fold[fold].append(item['path'])
        
        print("Arquivos por fold no cache:")
        for fold in sorted(cache_by_fold.keys()):
            print(f"  {fold}: {len(cache_by_fold[fold])} arquivos")
        
        # Verificar para cada iteração de CV
        print("\n" + "=" * 60)
        print("VERIFICANDO SPLIT POR ITERAÇÃO DE CV")
        print("=" * 60 + "\n")
        
        for test_idx, test_fold in enumerate(folds):
            val_fold = folds[(test_idx + 1) % len(folds)]
            train_folds = [f for f in folds if f != test_fold and f != val_fold]
            
            print(f"\nIteração {test_idx + 1}:")
            print(f"  Test:  {test_fold}")
            print(f"  Val:   {val_fold}")
            print(f"  Train: {train_folds}")
            
            # Contar arquivos
            test_files = set(cache_by_fold.get(test_fold, []))
            val_files = set(cache_by_fold.get(val_fold, []))
            train_files = set()
            for fold in train_folds:
                train_files.update(cache_by_fold.get(fold, []))
            
            print(f"  Contagem - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
            
            # Verificar overlaps
            overlap_train_val = train_files & val_files
            overlap_train_test = train_files & test_files
            overlap_val_test = val_files & test_files
            
            if overlap_train_val:
                print(f"  ⚠️  LEAK: Train ∩ Val = {len(overlap_train_val)} arquivos")
                for f in list(overlap_train_val)[:3]:
                    print(f"      {f}")
            
            if overlap_train_test:
                print(f"  ⚠️  LEAK: Train ∩ Test = {len(overlap_train_test)} arquivos")
            
            if overlap_val_test:
                print(f"  ⚠️  LEAK: Val ∩ Test = {len(overlap_val_test)} arquivos")
            
            if not (overlap_train_val or overlap_train_test or overlap_val_test):
                print(f"  ✓ Sem leakage detectado")
    
    # Verificar augmented data
    augmented_path = config.PROJECT_ROOT / "datasets" / "augmentation"
    augmented_metadata_path = augmented_path / "augmented_files_index.json"
    
    if os.path.exists(augmented_metadata_path):
        print("\n" + "=" * 60)
        print("VERIFICANDO AUGMENTED DATA")
        print("=" * 60)
        
        with open(augmented_metadata_path) as f:
            augmented_metadata = json.load(f)
        
        print(f"Total de arquivos augmentados: {len(augmented_metadata)}\n")
        
        # Agrupar por fold
        augmented_by_fold = {}
        for item in augmented_metadata:
            fold = item.get('fold')
            if fold not in augmented_by_fold:
                augmented_by_fold[fold] = []
            augmented_by_fold[fold].append(item['path'])
        
        print("Arquivos augmentados por fold:")
        for fold in sorted(augmented_by_fold.keys()):
            print(f"  {fold}: {len(augmented_by_fold[fold])} arquivos")
    
    # Verificar soundata original
    print("\n" + "=" * 60)
    print("VERIFICANDO SOUNDATA ORIGINAL")
    print("=" * 60)
    
    try:
        import soundata
        dataset = soundata.initialize("urbansound8k", data_home=dataset_path)
        all_clips = dataset.load_clips()
        
        by_fold = {}
        for clip_id, clip in all_clips.items():
            fold = f"fold{clip.fold}"
            if fold not in by_fold:
                by_fold[fold] = []
            by_fold[fold].append(clip_id)
        
        print(f"\nTotal de clips originais: {len(all_clips)}\n")
        print("Clips por fold no soundata:")
        for fold in sorted(by_fold.keys()):
            print(f"  {fold}: {len(by_fold[fold])} clips")
            
    except Exception as e:
        print(f"Erro ao carregar soundata: {e}")

if __name__ == "__main__":
    from trainer import Train
    
    FOLDS = [f"fold{i}" for i in range(1, 11)]
    DATA_PATH = "datasets"
    
    check_data_leakage(DATA_PATH, FOLDS)