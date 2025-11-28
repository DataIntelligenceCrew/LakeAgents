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
    准备测试数据：复制子表并生成 join/non-join pairs
    只处理在 omnimatch_datasets_dir 中存在的数据集
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
    
    join_pairs = []
    non_join_pairs = []
    
    print("Preparing test data...")
    
    for item in analysis_results:
        if item.get('status') != 'success' or 'result' not in item:
            continue
        
        dataset_name = item['dataset']
        
        # 只处理在 omnimatch2/datasets 中存在的数据集
        if dataset_name not in available_datasets:
            print(f"  Skipping {dataset_name}: not in omnimatch2/datasets")
            continue
        result = item['result']
        
        if 'candidate_table' not in result or 'non_candidate_table' not in result:
            continue
        
        ct_name = result['candidate_table'].get('name', 'Candidate_Features')
        nct_name = result['non_candidate_table'].get('name', 'NonCandidate_With_Target')
        join_columns = result.get('join_columns', [])
        
        if not join_columns:
            continue
        
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
        
        # 读取列名
        df_candidate = pd.read_csv(candidate_path, nrows=1)
        df_non_candidate = pd.read_csv(non_candidate_path, nrows=1)
        
        # 去除列名的尾随空格，与 csvs_to_dataframes 保持一致
        candidate_cols = [col.rstrip() for col in df_candidate.columns]
        non_candidate_cols = [col.rstrip() for col in df_non_candidate.columns]
        
        # 生成 pairs
        for col1 in candidate_cols:
            for col2 in non_candidate_cols:
                pair = ((candidate_table_file, col1), (non_candidate_table_file, col2))
                
                # 如果两个列都在 join_columns 中且列名匹配，则是 join pair
                # 注意：join_columns 中的列名也需要去除尾随空格
                join_columns_stripped = [jc.rstrip() for jc in join_columns]
                if col1 in join_columns_stripped and col2 in join_columns_stripped and col1 == col2:
                    join_pairs.append(pair)
                else:
                    non_join_pairs.append(pair)
    
    # 平衡数据
    if len(non_join_pairs) > len(join_pairs) * 10:
        random.seed(42)  # 使用 random.seed 而不是 np.random.seed
        sample_size = min(len(non_join_pairs), len(join_pairs) * 10)
        non_join_pairs = random.sample(non_join_pairs, sample_size)  # 使用 random.sample
    
    # 保存 pairs
    with open(os.path.join(test_matches_dir, "join_pairs.pickle"), 'wb') as f:
        pickle.dump(join_pairs, f)
    with open(os.path.join(test_matches_dir, "non_join_pairs.pickle"), 'wb') as f:
        pickle.dump(non_join_pairs, f)
    
    print(f"\nGenerated {len(join_pairs)} join pairs, {len(non_join_pairs)} non-join pairs")
    
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
    
    # 加载 column_ids
    with open(os.path.join(test_features_dir, "individual_features.pickle"), 'rb') as f:
        column_features = pickle.load(f)
    
    column_ids = {col: i for i, col in enumerate(column_features.keys())}
    id_to_column = {i: col for col, i in column_ids.items()}
    
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
        
        for col1 in df_candidate.columns:
            # 查找所有具有相同列名的列（不管表名），用于匹配训练数据中的 embedding
            col1_candidates = []
            for col_key, col_id in column_ids.items():
                table_name, col_name = col_key
                if col_name == col1:  # 列名匹配
                    col1_candidates.append((col_id, table_name))
            
            if not col1_candidates:
                # 如果找不到匹配的列，跳过
                continue
            
            for col2 in df_non_candidate.columns:
                # 只考虑列名匹配的情况（因为这是同一个数据集内的 join）
                if col1 != col2:
                    continue
                
                # 查找所有具有相同列名的列（不管表名）
                col2_candidates = []
                for col_key, col_id in column_ids.items():
                    table_name, col_name = col_key
                    if col_name == col2:  # 列名匹配
                        col2_candidates.append((col_id, table_name))
                
                if not col2_candidates:
                    continue
                
                # 计算所有候选对之间的相似度，取最高分
                best_score = 0
                best_pair = None
                for col1_id, col1_table in col1_candidates:
                    for col2_id, col2_table in col2_candidates:
                        try:
                            emb1 = embeddings[col1_id]
                            emb2 = embeddings[col2_id]
                            distance = torch.norm(emb1 - emb2).item()
                            score = 1.0 / (1.0 + distance)
                            if score > best_score:
                                best_score = score
                                best_pair = (col1_table, col2_table)
                        except (IndexError, KeyError) as e:
                            # 如果 embedding 索引超出范围，跳过
                            continue
                
                if best_score > 0 and (col1 not in best_scores or best_score > best_scores[col1][0]):
                    best_scores[col1] = (best_score, col2)
        
        # 按分数排序，选择 top join columns
        sorted_cols = sorted(best_scores.items(), key=lambda x: x[1][0], reverse=True)
        predicted_join_cols = [col for col, (score, _) in sorted_cols if score > 0.5]  # 阈值可调
        
        omnimatch_results[dataset_name] = predicted_join_cols
        
        # 比较结果
        llm_set = set(llm_join_columns)
        omnimatch_set = set(predicted_join_cols)
        
        comparison_results[dataset_name] = {
            'llm_join_columns': llm_join_columns,
            'omnimatch_join_columns': predicted_join_cols,
            'match': llm_set == omnimatch_set,
            'llm_only': list(llm_set - omnimatch_set),
            'omnimatch_only': list(omnimatch_set - llm_set),
            'common': list(llm_set & omnimatch_set)
        }
        
        print(f"\n{dataset_name}:")
        print(f"  LLM:        {llm_join_columns}")
        print(f"  OmniMatch:  {predicted_join_cols}")
        print(f"  Match:      {comparison_results[dataset_name]['match']}")
        if not comparison_results[dataset_name]['match']:
            print(f"  LLM only:   {comparison_results[dataset_name]['llm_only']}")
            print(f"  OmniMatch only: {comparison_results[dataset_name]['omnimatch_only']}")
    
    return omnimatch_results, comparison_results

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

if __name__ == "__main__":
    main()