#!/usr/bin/env python3
"""
准备 OmniMatch 测试数据
1. 从 subtables 复制 candidate_table 和 non_candidate_table 到 OmniMatch 测试目录
2. 基于 LLM 选出的 join columns 生成 join/non-join pairs
3. 运行 featurizer 提取特征
4. 运行 OmniMatch 预测
5. 提取并比较结果
"""

import os
import json
import pickle
import pandas as pd
import shutil
import subprocess
import sys
import glob
import torch
import numpy as np
import random

# 解决 MKL threading layer 兼容性问题
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def prepare_test_data(analysis_results_file, subtables_dir, output_dir, omnimatch_datasets_dir):
    """
    准备测试数据：复制子表并生成 table-level pairs
    只处理在 omnimatch_datasets_dir 中存在的数据集
    
    创建所有 table 两两的 pairs（所有 subtable 都要两两配对，不管是否来自同一个 original table）：
    - Positive: 同一个 dataset（original table）的 candidate 和 non-candidate，且该 dataset 可以成功 rejoin（在 verified_datasets 中）
    - Negative: 其他所有情况（不同 dataset、同一个 dataset 但类型相同、同一个 dataset 但不能成功 rejoin）
    """
    os.makedirs(output_dir, exist_ok=True)
    test_datasets_dir = os.path.join(output_dir, "test_datasets")
    test_matches_dir = os.path.join(output_dir, "test_matches")
    os.makedirs(test_datasets_dir, exist_ok=True)
    os.makedirs(test_matches_dir, exist_ok=True)
    
    # 获取 omnimatch2/datasets 中可用的数据集名称（去掉 .csv 后缀）
    available_datasets = set()
    if os.path.exists(omnimatch_datasets_dir):
        for filename in os.listdir(omnimatch_datasets_dir):
            if filename.endswith('.csv'):
                dataset_name = filename[:-4]  # 去掉 .csv 后缀
                available_datasets.add(dataset_name)
    
    print(f"Available datasets in omnimatch2/datasets: {sorted(available_datasets)}")
    
    # 加载分析结果
    with open(analysis_results_file, 'r') as f:
        analysis_results = json.load(f)
    
    # 存储所有 table 的信息
    # 格式：{table_file: {'dataset': dataset_name, 'type': 'candidate'|'non_candidate'}}
    all_tables_info = {}
    
    # 加载 verified_tables.json 获取成功 join 的 dataset 列表
    verified_tables_file = "/localdisk3/ytang49/opendata/verified_tables.json"
    verified_datasets = set()
    if os.path.exists(verified_tables_file):
        try:
            with open(verified_tables_file, 'r') as f:
                verified_data = json.load(f)
                # 支持两种格式：直接是列表，或者包含 "verified_tables" 字段
                if isinstance(verified_data, list):
                    verified_datasets = set(verified_data)
                elif isinstance(verified_data, dict) and 'verified_tables' in verified_data:
                    verified_datasets = set(verified_data['verified_tables'])
                else:
                    verified_datasets = set(verified_data)
            print(f"Loaded {len(verified_datasets)} verified datasets from {verified_tables_file}")
        except Exception as e:
            print(f"Warning: Failed to load verified_tables.json: {e}")
            print("  Will mark all table pairs as negative")
    else:
        print(f"Warning: {verified_tables_file} not found")
        print("  Will mark all table pairs as negative")
    
    # 存储 table-level 的 ground truth
    positive_table_pairs = set()
    negative_table_pairs = set()
    
    print("Preparing test data...")
    
    # 统计信息
    skipped_count = 0
    skipped_reasons = {}
    
    # 第一步：复制所有 tables 并记录信息
    for item in analysis_results:
        dataset_name = item.get('dataset', 'unknown')
        
        if item.get('status') != 'success' or 'result' not in item:
            skipped_count += 1
            reason = f"status={item.get('status')}, result={'missing' if 'result' not in item else 'present'}"
            skipped_reasons[dataset_name] = reason
            print(f"  Skipping {dataset_name}: {reason}")
            continue
        
        # 只处理在 omnimatch2/datasets 中存在的数据集
        if dataset_name not in available_datasets:
            skipped_count += 1
            skipped_reasons[dataset_name] = "not in available_datasets"
            print(f"  Skipping {dataset_name}: not in omnimatch2/datasets")
            continue
        
        result = item['result']
        
        if 'candidate_table' not in result or 'non_candidate_table' not in result:
            skipped_count += 1
            reason = f"missing candidate_table or non_candidate_table"
            skipped_reasons[dataset_name] = reason
            print(f"  Skipping {dataset_name}: {reason}")
            continue
        
        ct_name = result['candidate_table'].get('name', 'Candidate_Features')
        nct_name = result['non_candidate_table'].get('name', 'NonCandidate_With_Target')
        
        # 构建表文件名（OmniMatch 格式：dataset_name_table_name.csv）
        candidate_table_file = f"{dataset_name}_{ct_name}.csv"
        non_candidate_table_file = f"{dataset_name}_{nct_name}.csv"
        
        # 复制子表
        candidate_path = os.path.join(subtables_dir, dataset_name, f"{ct_name}.csv")
        non_candidate_path = os.path.join(subtables_dir, dataset_name, f"{nct_name}.csv")
        
        if not os.path.exists(candidate_path) or not os.path.exists(non_candidate_path):
            print(f"  Skipping {dataset_name}: subtable files not found")
            continue
        
        shutil.copy(candidate_path, os.path.join(test_datasets_dir, candidate_table_file))
        shutil.copy(non_candidate_path, os.path.join(test_datasets_dir, non_candidate_table_file))
        
        print(f"  Copied {dataset_name}: {candidate_table_file}, {non_candidate_table_file}")
        
        # 存储 table 信息
        all_tables_info[candidate_table_file] = {
            'dataset': dataset_name,
            'type': 'candidate'
        }
        all_tables_info[non_candidate_table_file] = {
            'dataset': dataset_name,
            'type': 'non_candidate'
        }
    
    # 第二步：创建所有 table 两两的 pairs
    # 所有 subtable 都要两两配对（组合，不重复），不管是否来自同一个 original table
    print("\nGenerating all table pairs...")
    table_files = list(all_tables_info.keys())
    print(f"Total tables: {len(table_files)}")
    print(f"Total pairs will be: {len(table_files) * (len(table_files) - 1) // 2} (combinations, excluding self-pairs)")
    
    for i, table1 in enumerate(table_files):
        info1 = all_tables_info[table1]
        dataset1 = info1['dataset']
        type1 = info1['type']
        
        for table2 in table_files[i+1:]:  # 只与后面的 table 配对，避免重复 (A,B) 和 (B,A)
            info2 = all_tables_info[table2]
            dataset2 = info2['dataset']
            type2 = info2['type']
            
            # 使用排序后的 tuple 作为 key（(A,B) 和 (B,A) 被视为同一个 pair）
            # 生成所有 C(n,2) = n*(n-1)/2 个组合（排除自己和自己配对）
            table_pair_key = tuple(sorted([table1, table2]))
            
            # 判断是 positive 还是 negative
            # Positive: 同一个 dataset（original table）的 candidate 和 non_candidate，且该 dataset 可以成功 rejoin（在 verified_datasets 中）
            if (dataset1 == dataset2 and 
                dataset1 in verified_datasets and
                ((type1 == 'candidate' and type2 == 'non_candidate') or 
                 (type1 == 'non_candidate' and type2 == 'candidate'))):
                # 可以成功 rejoin，标记为 positive
                positive_table_pairs.add(table_pair_key)
            else:
                # 其他所有情况都是 negative：
                # - 不同 dataset 的 table pairs
                # - 同一个 dataset 但都是 candidate 或都是 non_candidate
                # - 同一个 dataset 的 candidate 和 non_candidate 但该 dataset 不能成功 rejoin（不在 verified_datasets 中）
                negative_table_pairs.add(table_pair_key)
    
    # 保存 table-level ground truth
    with open(os.path.join(test_matches_dir, "positive_table_pairs.pickle"), 'wb') as f:
        pickle.dump(positive_table_pairs, f)
    with open(os.path.join(test_matches_dir, "negative_table_pairs.pickle"), 'wb') as f:
        pickle.dump(negative_table_pairs, f)
    
    # 创建空的 column-level pairs 文件（featurizer 需要这些文件，但我们不需要计算 pairwise features）
    # 我们只需要 individual_features.pickle
    empty_join_pairs = []
    empty_non_join_pairs = []
    with open(os.path.join(test_matches_dir, "join_pairs.pickle"), 'wb') as f:
        pickle.dump(empty_join_pairs, f)
    with open(os.path.join(test_matches_dir, "non_join_pairs.pickle"), 'wb') as f:
        pickle.dump(empty_non_join_pairs, f)
    
    print(f"\nGenerated {len(positive_table_pairs)} positive table pairs, {len(negative_table_pairs)} negative table pairs")
    print(f"Created empty column-level pairs files (only need individual_features)")
    
    return test_datasets_dir, test_matches_dir

def run_featurizer(omnimatch_dir, test_datasets_dir, test_matches_dir, test_features_dir):
    """运行 featurizer 提取测试数据特征"""
    print("\n" + "=" * 60)
    print("Running Featurizer on test data...")
    print("=" * 60)
    
    featurizer_config = f"""[PATHS]
dataset_path: {test_datasets_dir}
features_path: {test_features_dir}
join_pairs_file: {test_matches_dir}/join_pairs.pickle
non_join_pairs_file: {test_matches_dir}/non_join_pairs.pickle
embeddings_file: 

[FEATURES]
jaccard_frequent: True
value_embeddings: False
value_distribution: True
jaccard_containment: True

[OTHER]
compute_column_features: True
"""
    
    config_path = os.path.join(omnimatch_dir, "config_files", "featurizer_config_test.ini")
    with open(config_path, 'w') as f:
        f.write(featurizer_config)
    
    # 不使用 capture_output，让输出实时显示
    result = subprocess.run([
        sys.executable,
        os.path.join(omnimatch_dir, "src", "featurizer.py"),
        "-cf", config_path
    ])  # 移除 capture_output=True，让输出实时显示
    
    if result.returncode != 0:
        raise RuntimeError("Featurizer failed")
    
    print("Featurizer completed successfully")
    return config_path

def run_omnimatch_prediction(omnimatch_dir, test_features_dir, results_dir):
    """运行 OmniMatch 预测（使用训练好的模型）"""
    print("\n" + "=" * 60)
    print("Running OmniMatch prediction...")
    print("=" * 60)
    
    # 更新配置文件，使用测试数据
    predictor_config = f"""[PATHS]
train_datasets_path: /localdisk3/ytang49/opendata/omnimatch/datasets/city_government/train_tables
train_features_path: /localdisk3/ytang49/opendata/omnimatch/assets/features/city_government/train_tables/
train_node_features: /localdisk3/ytang49/opendata/omnimatch/assets/features/city_government/train_tables/individual_features.pickle
test_features_path: {test_features_dir}
test_node_features: {test_features_dir}/individual_features.pickle
results_path: {results_dir}
samples_path: /localdisk3/ytang49/opendata/omnimatch/assets/samples/city_government
sampled_datasets: 

[PARAMETERS]
benchmark: city_government
graph_construction: topk
model_loss: rgcn_margin
k: 3
number_of_datasets: 2
number_of_sources: 20
dimension: 256
epochs: 30
learning_rate: 0.001
margin: 0.5
norm: 2

[FEATURES]
jaccard_frequent: True
value_embeddings: False
value_distribution: True
jaccard_containment: True

[SAVEFILES]
write_embeddings: True
write_results: True
"""
    
    config_path = os.path.join(omnimatch_dir, "config_files", "omnimatch_predictors_config_test.ini")
    with open(config_path, 'w') as f:
        f.write(predictor_config)
    
    # 设置环境变量以解决 MKL threading layer 兼容性问题
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    # 或者使用：env['MKL_THREADING_LAYER'] = 'GNU'
    
    # 不使用 capture_output，让输出实时显示
    result = subprocess.run([
        sys.executable,
        os.path.join(omnimatch_dir, "src", "omnimatch_predictors.py"),
        "-cf", config_path
    ], env=env)  # 传递修改后的环境变量
    
    if result.returncode != 0:
        raise RuntimeError("OmniMatch failed")
    
    print("OmniMatch prediction completed successfully")
    return config_path

def extract_omnimatch_join_columns(test_features_dir, test_datasets_dir, results_dir, analysis_results_file):
    """
    从 OmniMatch 结果中提取预测的 join columns
    并与 LLM 结果比较
    """
    print("\n" + "=" * 60)
    print("Extracting OmniMatch predictions and comparing with LLM...")
    print("=" * 60)
    
    # 加载 embeddings
    embeddings_pattern = os.path.join(results_dir, "rgcn_embeddings_*.pickle")
    embeddings_files = glob.glob(embeddings_pattern)
    if not embeddings_files:
        raise FileNotFoundError(f"No embeddings file found matching {embeddings_pattern}")
    embeddings_file = max(embeddings_files, key=os.path.getctime)
    print(f"Loading embeddings from {embeddings_file}")
    
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # 转换为 tensor
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    elif isinstance(embeddings, torch.Tensor):
        pass
    else:
        embeddings = torch.stack([torch.tensor(e) if isinstance(e, np.ndarray) else e for e in embeddings])
    
    # 关键修改：使用 training data 的 column_ids，因为 embeddings 是基于 training graph 计算的
    train_node_features_path = "/localdisk3/ytang49/opendata/omnimatch/assets/features/city_government/train_tables/individual_features.pickle"
    
    if not os.path.exists(train_node_features_path):
        raise FileNotFoundError(f"Training node features not found: {train_node_features_path}")
    
    # 加载 training data 的 column_ids（embeddings 的索引对应这个）
    with open(train_node_features_path, 'rb') as f:
        train_column_features = pickle.load(f)
    
    # 使用与 get_column_ids 相同的逻辑
    train_column_ids = {}
    for i, (col, features) in enumerate(train_column_features.items()):
        # 注意：get_column_ids 中使用了 rstrip()
        col_key = (col[0], col[1].rstrip())
        train_column_ids[col_key] = i
    
    print(f"Loaded {len(train_column_ids)} training column IDs")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 加载 test data 的 column features（用于获取列名）
    with open(os.path.join(test_features_dir, "individual_features.pickle"), 'rb') as f:
        test_column_features = pickle.load(f)
    
    # 加载 LLM 结果
    with open(analysis_results_file, 'r') as f:
        analysis_results = json.load(f)
    
    # 为每个数据集提取预测的 join columns
    omnimatch_results = {}
    comparison_results = {}
    
    for item in analysis_results:
        if item.get('status') != 'success' or 'result' not in item:
            continue
        
        dataset_name = item['dataset']
        result = item['result']
        
        if 'candidate_table' not in result or 'non_candidate_table' not in result:
            continue
        
        ct_name = result['candidate_table'].get('name', 'Candidate_Features')
        nct_name = result['non_candidate_table'].get('name', 'NonCandidate_With_Target')
        llm_join_columns = result.get('join_columns', [])
        
        # 对 LLM 的列名也进行 rstrip() 处理，与 featurizer 保持一致
        llm_join_columns = [col.rstrip() for col in llm_join_columns]
        
        candidate_table_file = f"{dataset_name}_{ct_name}.csv"
        non_candidate_table_file = f"{dataset_name}_{nct_name}.csv"
        
        # 加载表获取列名
        candidate_path = os.path.join(test_datasets_dir, candidate_table_file)
        non_candidate_path = os.path.join(test_datasets_dir, non_candidate_table_file)
        
        if not os.path.exists(candidate_path) or not os.path.exists(non_candidate_path):
            continue
        
        df_candidate = pd.read_csv(candidate_path, nrows=1)
        df_non_candidate = pd.read_csv(non_candidate_path, nrows=1)
        
        # 去除列名的尾随空格，与 featurizer 保持一致
        df_candidate.columns = [col.rstrip() for col in df_candidate.columns]
        df_non_candidate.columns = [col.rstrip() for col in df_non_candidate.columns]
        
        # 计算所有列对的分数
        best_scores = {}  # {col_name: (score, matched_col)}
        
        # 为每个 candidate 列找到对应的 embedding（只根据 column name，忽略 table name）
        col1_to_emb = {}
        for col1 in df_candidate.columns:
            # 只根据 column name 查找，忽略 table name
            for (train_table, train_col), col_id in train_column_ids.items():
                if train_col == col1:  # 只比较 column name
                    try:
                        if col_id < len(embeddings):
                            col1_to_emb[col1] = embeddings[col_id]
                            break  # 找到第一个匹配就停止
                    except (IndexError, KeyError):
                        pass
        
        # 为每个 non-candidate 列找到对应的 embedding（只根据 column name，忽略 table name）
        col2_to_emb = {}
        for col2 in df_non_candidate.columns:
            # 只根据 column name 查找，忽略 table name
            for (train_table, train_col), col_id in train_column_ids.items():
                if train_col == col2:  # 只比较 column name
                    try:
                        if col_id < len(embeddings):
                            col2_to_emb[col2] = embeddings[col_id]
                            break  # 找到第一个匹配就停止
                    except (IndexError, KeyError):
                        pass
        
        # 计算所有列对的 embedding 相似度（不再要求列名匹配）
        for col1, emb1 in col1_to_emb.items():
            for col2, emb2 in col2_to_emb.items():
                try:
                    distance = torch.norm(emb1 - emb2).item()
                    score = 1.0 / (1.0 + distance)
                    
                    # 记录每个 candidate 列的最佳匹配
                    if col1 not in best_scores or score > best_scores[col1][0]:
                        best_scores[col1] = (score, col2)
                except Exception as e:
                    continue
        
        # 按分数排序，选择 top join columns
        sorted_cols = sorted(best_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        # 根据 LLM 的 join column 数量，选择 top n
        n_llm = len(llm_join_columns)
        if n_llm > 0:
            # 选择 top n，但只保留分数 > 0.5 的
            predicted_join_cols = [col for col, (score, _) in sorted_cols[:n_llm] if score > 0.5]
        else:
            # 如果 LLM 没有 join columns，OmniMatch 也不选择
            predicted_join_cols = []
        
        omnimatch_results[dataset_name] = predicted_join_cols
        
        # 比较结果
        # 确保两个集合中的列名都经过 rstrip() 处理
        llm_set = set(llm_join_columns)  # 已经 rstrip() 过了
        omnimatch_set = set(predicted_join_cols)  # 已经 rstrip() 过了
        
        # 计算重叠比例
        intersection = llm_set & omnimatch_set
        union = llm_set | omnimatch_set
        
        # 重叠比例：重叠的列数 / LLM 的列数
        overlap_ratio = len(intersection) / len(llm_set) if len(llm_set) > 0 else 0.0
        # Jaccard 相似度：重叠的列数 / 并集的列数
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        comparison_results[dataset_name] = {
            'llm_join_columns': llm_join_columns,
            'omnimatch_join_columns': predicted_join_cols,
            'llm_count': len(llm_join_columns),
            'omnimatch_count': len(predicted_join_cols),
            'match': llm_set == omnimatch_set,
            'overlap_ratio': overlap_ratio,  # 重叠列数 / LLM 列数
            'jaccard_similarity': jaccard_similarity,  # 重叠列数 / 并集列数
            'intersection_count': len(intersection),
            'llm_only': list(llm_set - omnimatch_set),
            'omnimatch_only': list(omnimatch_set - llm_set),
            'common': list(intersection)
        }
        
        print(f"\n{dataset_name}:")
        print(f"  LLM:        {llm_join_columns} (n={len(llm_join_columns)})")
        print(f"  OmniMatch:  {predicted_join_cols} (n={len(predicted_join_cols)})")
        print(f"  Overlap:    {len(intersection)}/{len(llm_set)} = {overlap_ratio:.2%}")
        print(f"  Jaccard:    {jaccard_similarity:.2%}")
        print(f"  Match:      {comparison_results[dataset_name]['match']}")
        if not comparison_results[dataset_name]['match']:
            print(f"  LLM only:   {comparison_results[dataset_name]['llm_only']}")
            print(f"  OmniMatch only: {comparison_results[dataset_name]['omnimatch_only']}")
    
    return omnimatch_results, comparison_results

def compute_table_level_confusion_matrix(test_features_dir, test_datasets_dir, test_matches_dir, results_dir, analysis_results_file, threshold=0.5, top_k=2):
    """
    计算 table-level 的 TP/TN/FP/FN
    
    定义：
    - TP: Ground truth positive，且预测的 top-k column pairs 100% 覆盖了 ground truth 的 join columns
    - TN: Ground truth negative，且预测 negative（没有选出 join column）
    - FP: Ground truth negative 但预测 positive，或 ground truth positive 但预测的 join columns 与 ground truth 不匹配
    - FN: Ground truth positive，但预测 negative（没有选出 join column）
    
    对于每个 table pair：
    1. 计算所有 column pair 的 similarity
    2. 选择 top-k 的 column pairs（按 similarity 降序排序），但要求：
       - 四个 column 都不重复（T1 的列不重复，T2 的列也不重复）
       - 如果有 overlap，选择 similarity 更高的（通过排序实现）
    3. 如果这 top-k 个 pairs 的 similarity 都 > threshold，则预测为可以 join（positive）
    4. 否则预测为不能 join（negative）
    
    使用 OmniMatch 的 embedding distance 计算 similarity
    """
    print("\n" + "=" * 60)
    print(f"Computing table-level confusion matrix with threshold={threshold}, top_k={top_k}...")
    print("=" * 60)
    
    # 加载 table-level ground truth
    with open(os.path.join(test_matches_dir, "positive_table_pairs.pickle"), 'rb') as f:
        positive_table_pairs = pickle.load(f)
    with open(os.path.join(test_matches_dir, "negative_table_pairs.pickle"), 'rb') as f:
        negative_table_pairs = pickle.load(f)
    
    print(f"Loaded {len(positive_table_pairs)} positive table pairs, {len(negative_table_pairs)} negative table pairs")
    
    # 加载 ground truth join columns（从 analysis_results 中获取）
    ground_truth_join_columns = {}  # {(table1, table2): set of join column names}
    if analysis_results_file and os.path.exists(analysis_results_file):
        with open(analysis_results_file, 'r') as f:
            analysis_results = json.load(f)
        
        for item in analysis_results:
            if item.get('status') != 'success' or 'result' not in item:
                continue
            
            dataset_name = item['dataset']
            result = item['result']
            
            if 'candidate_table' not in result or 'non_candidate_table' not in result:
                continue
            
            ct_name = result['candidate_table'].get('name', 'Candidate_Features')
            nct_name = result['non_candidate_table'].get('name', 'NonCandidate_With_Target')
            
            candidate_table_file = f"{dataset_name}_{ct_name}.csv"
            non_candidate_table_file = f"{dataset_name}_{nct_name}.csv"
            
            # 获取 ground truth join columns
            llm_join_columns = result.get('join_columns', [])
            llm_join_columns = [col.rstrip() for col in llm_join_columns]
            
            # 创建 table pair key（排序后的 tuple，与创建 pairs 时的格式一致）
            table_pair_key = tuple(sorted([candidate_table_file, non_candidate_table_file]))
            ground_truth_join_columns[table_pair_key] = set(llm_join_columns)
    
    print(f"Loaded ground truth join columns for {len(ground_truth_join_columns)} table pairs")
    
    # 加载 embeddings
    embeddings_pattern = os.path.join(results_dir, "rgcn_embeddings_*.pickle")
    embeddings_files = glob.glob(embeddings_pattern)
    if not embeddings_files:
        raise FileNotFoundError(f"No embeddings file found matching {embeddings_pattern}")
    embeddings_file = max(embeddings_files, key=os.path.getctime)
    print(f"Loading embeddings from {embeddings_file}")
    
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # 转换为 tensor
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    elif isinstance(embeddings, torch.Tensor):
        pass
    else:
        embeddings = torch.stack([torch.tensor(e) if isinstance(e, np.ndarray) else e for e in embeddings])
    
    # 加载 training data 的 column_ids
    train_node_features_path = "/localdisk3/ytang49/opendata/omnimatch/assets/features/city_government/train_tables/individual_features.pickle"
    if not os.path.exists(train_node_features_path):
        raise FileNotFoundError(f"Training node features not found: {train_node_features_path}")
    
    with open(train_node_features_path, 'rb') as f:
        train_column_features = pickle.load(f)
    
    train_column_ids = {}
    for i, (col, features) in enumerate(train_column_features.items()):
        col_key = (col[0], col[1].rstrip())
        train_column_ids[col_key] = i
    
    # 加载 test data 的 column features
    with open(os.path.join(test_features_dir, "individual_features.pickle"), 'rb') as f:
        test_column_features = pickle.load(f)
    
    # 构建 column 到 embedding 的映射
    col_to_emb = {}
    for col, features in test_column_features.items():
        col_key = (col[0], col[1].rstrip())
        # 在 training data 中查找匹配的 embedding（只根据 column name）
        for (train_table, train_col), col_id in train_column_ids.items():
            if train_col == col_key[1]:  # 只比较 column name
                try:
                    if col_id < len(embeddings):
                        col_to_emb[col_key] = embeddings[col_id]
                        break
                except (IndexError, KeyError):
                    pass
    
    print(f"Found embeddings for {len(col_to_emb)} columns")
    
    # 为每个 table pair 计算最大 similarity
    all_table_pairs = positive_table_pairs | negative_table_pairs
    table_pair_results = {}
    
    for table_pair_key in all_table_pairs:
        table1, table2 = table_pair_key
        
        # 获取两个 table 的所有列
        cols1 = []
        cols2 = []
        
        for col_key in col_to_emb.keys():
            if col_key[0] == table1:
                cols1.append(col_key)
            elif col_key[0] == table2:
                cols2.append(col_key)
        
        # 计算所有列对的 similarity
        all_pairs_similarities = []
        
        for col1_key in cols1:
            for col2_key in cols2:
                try:
                    emb1 = col_to_emb[col1_key]
                    emb2 = col_to_emb[col2_key]
                    # 使用 OmniMatch 的方式计算 distance（L2 norm）
                    distance = torch.norm(emb1 - emb2).item()
                    # 转换为 similarity score（与当前代码保持一致）
                    similarity = 1.0 / (1.0 + distance)
                    all_pairs_similarities.append({
                        'col1': col1_key,
                        'col2': col2_key,
                        'similarity': similarity
                    })
                except Exception as e:
                    continue
        
        # 按 similarity 降序排序
        all_pairs_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 选择 top-k 的 column pairs，确保四个 column 都不重复
        # 要求：T1 的列不重复，T2 的列也不重复
        top_k_pairs = []
        used_cols1 = set()  # T1 中已使用的列
        used_cols2 = set()  # T2 中已使用的列
        
        for pair in all_pairs_similarities:
            col1_key = pair['col1']
            col2_key = pair['col2']
            col1_name = col1_key[1]  # column name from table1
            col2_name = col2_key[1]  # column name from table2
            
            # 检查是否有列重复
            if col1_name not in used_cols1 and col2_name not in used_cols2:
                # 没有重复，可以添加
                top_k_pairs.append(pair)
                used_cols1.add(col1_name)
                used_cols2.add(col2_name)
                
                # 如果已经找到 top_k 个 pairs，停止
                if len(top_k_pairs) >= top_k:
                    break
        
        # 判断是否所有 top-k pairs 的 similarity 都 > threshold
        if len(top_k_pairs) >= top_k:
            all_above_threshold = all(p['similarity'] > threshold for p in top_k_pairs)
        else:
            # 如果 column pairs 数量不足 top_k，则不能预测为 positive
            all_above_threshold = False
        
        # 筛选 similarity > threshold 的 pairs（用于显示）
        predicted_positive_pairs = [
            p for p in all_pairs_similarities if p['similarity'] > threshold
        ]
        
        table_pair_results[table_pair_key] = {
            'table1': table1,
            'table2': table2,
            'total_column_pairs': len(all_pairs_similarities),
            'top_k_pairs': top_k_pairs,
            'all_above_threshold': all_above_threshold,
            'predicted_positive_count': len(predicted_positive_pairs),
            'predicted_positive_pairs': predicted_positive_pairs
        }
    
    # 计算混淆矩阵
    # 定义（根据标准定义）：
    # - TP: Ground truth positive，且预测的 top-k column pairs 100% 覆盖了 ground truth 的 join columns
    # - TN: Ground truth negative，且预测 negative（OmniMatch 没有选出 join column，test sample 也不存在 join column）
    # - FP: (1) Ground truth negative，但预测 positive（没有 label join column，但是 OmniMatch 选出来了 join column）
    #       (2) Ground truth positive，但预测的 join columns 与 ground truth 不匹配（选了 join column，但是跟 label 的不一样，没选对）
    # - FN: Ground truth positive，但预测 negative（有 label join column，但 OmniMatch 没选出来）
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for table_pair_key in all_table_pairs:
        is_positive_gt = table_pair_key in positive_table_pairs
        is_positive_pred = table_pair_results[table_pair_key]['all_above_threshold']
        top_k_pairs = table_pair_results[table_pair_key]['top_k_pairs']
        
        if is_positive_gt:
            # Ground truth positive（有 label join column）
            if is_positive_pred:
                # 预测也是 positive（OmniMatch 选出了 join column），需要检查是否 100% 覆盖了 ground truth join columns
                if table_pair_key in ground_truth_join_columns:
                    gt_join_cols = ground_truth_join_columns[table_pair_key]
                    # 提取预测的 join columns（从 top-k pairs 中）
                    # 对于每个 pair，如果两个 column name 相同，则认为这是一个 join column
                    predicted_join_cols = set()
                    for pair in top_k_pairs:
                        col1_name = pair['col1'][1].rstrip()
                        col2_name = pair['col2'][1].rstrip()
                        # 如果两个 column name 相同，则认为这是一个 join column
                        if col1_name == col2_name:
                            predicted_join_cols.add(col1_name)
                    
                    # 检查是否 100% 覆盖了 ground truth join columns
                    # TP 要求：预测的 join columns 必须完全覆盖 ground truth 的 join columns
                    if gt_join_cols.issubset(predicted_join_cols) and len(predicted_join_cols) >= len(gt_join_cols):
                        tp += 1
                    else:
                        # 预测了 join column，但没有 100% 覆盖 ground truth（或选错了）
                        fp += 1
                else:
                    # 没有 ground truth join columns 信息，但预测了 positive
                    # 保守处理：如果预测 positive 且 ground truth positive，算作 TP
                    tp += 1
            else:
                # 预测 negative（没有选出 join column），但 ground truth positive（有 label join column）
                # 根据标准定义：FN = 实际是 positive 但预测是 negative（假阴性）
                fn += 1
        else:
            # Ground truth negative（没有 label join column）
            if is_positive_pred:
                # 预测 positive（OmniMatch 选出了 join column），但 ground truth negative（没有 label join column）
                # 根据标准定义：FP = 实际是 negative 但预测是 positive
                fp += 1
            else:
                # 预测 negative（没有选出 join column），ground truth negative（没有 label join column）
                tn += 1
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # 总结所有 TP 案例（先收集，稍后输出）
    tp_cases = []
    for table_pair_key in all_table_pairs:
        is_positive_gt = table_pair_key in positive_table_pairs
        is_positive_pred = table_pair_results[table_pair_key]['all_above_threshold']
        if is_positive_gt and is_positive_pred:
            tp_cases.append((table_pair_key, table_pair_results[table_pair_key]))
    
    # 输出每个 table pair 的详细信息
    print(f"\n{'=' * 60}")
    print("All Table Pair Details:")
    print(f"{'=' * 60}")
    for table_pair_key, info in sorted(table_pair_results.items()):
        is_positive_gt = table_pair_key in positive_table_pairs
        is_positive_pred = info['all_above_threshold']
        status = "✓" if is_positive_gt == is_positive_pred else "✗"
        
        print(f"\n{status} {info['table1']} <-> {info['table2']}")
        print(f"  Ground truth: {'Positive' if is_positive_gt else 'Negative'}")
        print(f"  Prediction: {'Positive' if is_positive_pred else 'Negative'}")
        print(f"  Total column pairs: {info['total_column_pairs']}, Predicted positive: {info['predicted_positive_count']}")
        
        if info['top_k_pairs']:
            print(f"  Top {len(info['top_k_pairs'])} column pairs:")
            for i, p in enumerate(info['top_k_pairs']):
                above_threshold = "✓" if p['similarity'] > threshold else "✗"
                print(f"    {i+1}. {above_threshold} {p['col1'][1]} <-> {p['col2'][1]}: {p['similarity']:.3f}")
        
        if not info['all_above_threshold'] and info['top_k_pairs']:
            print(f"  → Not all top-{top_k} pairs above threshold {threshold}, predicted as Negative")
    
    # 最后输出汇总信息
    print(f"\n{'=' * 60}")
    print("Table-Level Confusion Matrix Summary:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Total: {tp + tn + fp + fn}")
    print(f"\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
    print(f"{'=' * 60}")
    
    # 输出 TP 案例汇总
    print(f"\n{'=' * 60}")
    print(f"True Positive (TP) Cases Summary ({len(tp_cases)} cases):")
    print(f"{'=' * 60}")
    for i, (table_pair_key, info) in enumerate(tp_cases, 1):
        print(f"\n{i}. {info['table1']} <-> {info['table2']}")
        print(f"   Total column pairs: {info['total_column_pairs']}, Predicted positive: {info['predicted_positive_count']}")
        if info['top_k_pairs']:
            print(f"   Top {len(info['top_k_pairs'])} column pairs:")
            for j, p in enumerate(info['top_k_pairs']):
                print(f"      {j+1}. {p['col1'][1]} <-> {p['col2'][1]}: {p['similarity']:.3f}")
    
    # 收集所有 TP 案例的详细信息
    tp_cases_details = []
    for table_pair_key in all_table_pairs:
        is_positive_gt = table_pair_key in positive_table_pairs
        is_positive_pred = table_pair_results[table_pair_key]['all_above_threshold']
        if is_positive_gt and is_positive_pred:
            info = table_pair_results[table_pair_key]
            tp_cases_details.append({
                'table1': info['table1'],
                'table2': info['table2'],
                'total_column_pairs': info['total_column_pairs'],
                'predicted_positive_count': info['predicted_positive_count'],
                'top_k_pairs': [
                    {
                        'col1': p['col1'][1],
                        'col2': p['col2'][1],
                        'similarity': float(p['similarity'])
                    }
                    for p in info['top_k_pairs']
                ]
            })
    
    return {
        'threshold': threshold,
        'top_k': top_k,
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': tp + tn + fp + fn
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        },
        'tp_cases_summary': {
            'count': len(tp_cases_details),
            'cases': tp_cases_details
        },
        'table_pair_results': table_pair_results
    }

def main():
    """主函数"""
    # 配置路径
    analysis_results_file = "/localdisk3/ytang49/opendata/analysis_results_optimized.json"
    subtables_dir = "/localdisk3/ytang49/opendata/subtables"
    omnimatch_dir = "/localdisk3/ytang49/opendata/omnimatch"
    omnimatch_datasets_dir = "/localdisk3/ytang49/opendata/omnimatch/data/omnimatch2/datasets"
    output_dir = "/localdisk3/ytang49/opendata/omnimatch/data/my_test_data"
    results_dir = "/localdisk3/ytang49/opendata/omnimatch/results_my_test"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    test_features_dir = os.path.join(output_dir, "test_features")
    os.makedirs(test_features_dir, exist_ok=True)
    
    print("=" * 60)
    print("OmniMatch Test Data Preparation and Prediction")
    print("=" * 60)
    
    # 1. 准备测试数据
    print("\nStep 1: Preparing test data...")
    test_datasets_dir, test_matches_dir = prepare_test_data(
        analysis_results_file, subtables_dir, output_dir, omnimatch_datasets_dir
    )
    
    # 2. 运行 featurizer
    print("\nStep 2: Running featurizer...")
    run_featurizer(omnimatch_dir, test_datasets_dir, test_matches_dir, test_features_dir)
    
    # 3. 运行 OmniMatch 预测
    print("\nStep 3: Running OmniMatch prediction...")
    run_omnimatch_prediction(omnimatch_dir, test_features_dir, results_dir)
    
    # 4. 提取并比较结果
    print("\nStep 4: Extracting and comparing results...")
    omnimatch_results, comparison_results = extract_omnimatch_join_columns(
        test_features_dir, test_datasets_dir, results_dir, analysis_results_file
    )
    
    # 5. 保存结果
    output_file = os.path.join(output_dir, "omnimatch_vs_llm_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_file}")
    print(f"{'=' * 60}")
    
    # 统计
    total = len(comparison_results)
    matches = sum(1 for r in comparison_results.values() if r['match'])
    print(f"\nSummary: {matches}/{total} datasets have matching join columns")
    
    # 6. 计算 table-level 混淆矩阵
    print("\nStep 6: Computing table-level confusion matrix...")
    confusion_matrix_results = compute_table_level_confusion_matrix(
        test_features_dir, test_datasets_dir, test_matches_dir, results_dir, 
        analysis_results_file=analysis_results_file, threshold=0.5
    )
    
    # 保存 table-level 混淆矩阵结果
    confusion_matrix_file = os.path.join(output_dir, "table_level_confusion_matrix.json")
    # 将结果转换为可序列化的格式
    serializable_results = {
        'threshold': confusion_matrix_results['threshold'],
        'top_k': confusion_matrix_results['top_k'],
        'confusion_matrix': confusion_matrix_results['confusion_matrix'],
        'metrics': confusion_matrix_results['metrics'],
        'tp_cases_summary': confusion_matrix_results['tp_cases_summary'],
        'table_pair_results': {
            f"{info['table1']}<->{info['table2']}": {
                'table1': info['table1'],
                'table2': info['table2'],
                'total_column_pairs': info['total_column_pairs'],
                'all_above_threshold': info['all_above_threshold'],
                'predicted_positive_count': info['predicted_positive_count'],
                'top_k_pairs': [
                    {
                        'col1': str(p['col1']),
                        'col2': str(p['col2']),
                        'similarity': float(p['similarity'])
                    }
                    for p in info['top_k_pairs']
                ],
                'predicted_positive_pairs': [
                    {
                        'col1': str(p['col1']),
                        'col2': str(p['col2']),
                        'similarity': float(p['similarity'])
                    }
                    for p in info['predicted_positive_pairs']
                ]
            }
            for info in confusion_matrix_results['table_pair_results'].values()
        }
    }
    
    with open(confusion_matrix_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nTable-level confusion matrix results saved to {confusion_matrix_file}")

if __name__ == "__main__":
    main()