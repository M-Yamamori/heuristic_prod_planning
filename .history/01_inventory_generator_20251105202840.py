#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初期在庫生成・調整モジュール
データフォルダ内の全CSVファイルに対して初期在庫を生成・調整し、CSVファイルとして出力する
"""

import os
import glob
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import japanize_matplotlib
from typing import List, Tuple, Dict

# コストとペナルティの係数
在庫コスト単価 = 180
残業コスト単価 = 66.7
段替えコスト単価 = 400
出荷遅れコスト単価 = 500

定時 = 8 * 60 * 2
最大残業時間 = 2 * 60 * 2
段替え時間 = 30

def read_production_data(file_path: str) -> Tuple[List, List, List, List, List, List]:
    """CSVファイルから生産データを読み込む関数"""
    
    # 収容数辞書を読み込み
    収容数辞書 = {}
    with open('収容数.csv', 'r', encoding='shift-jis') as capacity_file:
        capacity_reader = csv.reader(capacity_file)
        capacity_header = next(capacity_reader)
        for row in capacity_reader:
            if len(row) >= 2 and row[1].strip():
                品番 = row[0]
                収容数 = int(float(row[1]))
                収容数辞書[品番] = 収容数

    with open(file_path, 'r', encoding='shift-jis') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        品番リスト = []
        出荷数リスト = []
        収容数リスト = []
        サイクルタイムリスト = []
        込め数リスト = []
        
        # 期間数（日数）を定義
        期間数 = 20
        
        rows = list(reader)
        for row in rows:
            if len(row) == 0:
                continue
            
            # 個数を取得
            total_quantity = int(row[header.index("個数")])
            
            # 個数が200未満の場合はスキップ
            if total_quantity < 200:
                continue
            
            # 1日あたりの出荷数を計算（総期間で割る）
            daily_quantity = total_quantity / 期間数
            
            品番リスト.append(row[header.index("素材品番")])
            出荷数リスト.append(daily_quantity)
            込め数リスト.append(int(float(row[header.index("込数")])))
            
            cycle_time_per_unit = float(row[header.index("サイクルタイム")]) / 60
            サイクルタイムリスト.append(cycle_time_per_unit)
            
            収容数リスト.append(収容数辞書.get(row[header.index("素材品番")], 80))
    
    # 出荷数に基づいて初期在庫量（処理前）をランダムに設定
    初期在庫量リスト = []
    for shipment in 出荷数リスト:
        random_inventory = random.randint(int(shipment * 3), int(shipment * 5))
        初期在庫量リスト.append(random_inventory)
        
    return 品番リスト, 出荷数リスト, 収容数リスト, サイクルタイムリスト, 込め数リスト, 初期在庫量リスト

def simulate_production_schedule(initial_inventory: List[float], 
                               出荷数リスト: List[float],
                               サイクルタイムリスト: List[float],
                               込め数リスト: List[int],
                               期間: int = 20) -> List[List[float]]:
    """生産スケジュールをシミュレートする関数"""
    品番数 = len(initial_inventory)
    inventory = initial_inventory[:]
    inventory_history = [[] for _ in range(品番数)]
    
    daily_regular_time = 8 * 60 * 2  # 1日の通常稼働時間（分）
    max_daily_overtime = 2 * 60 * 2  # 1日の最大残業時間（分）
    max_daily_work_time = daily_regular_time + max_daily_overtime
    
    for t in range(期間):
        daily_production_time = 0
        daily_setup_count = 0
        
        # 各品番の需要を処理
        for i in range(品番数):
            demand = 出荷数リスト[i]
            inventory[i] -= demand
            
            # 在庫が不足する場合は生産
            if inventory[i] < 0:
                shortage = abs(inventory[i])
                # 生産量を決定（不足分を補う）
                production = shortage
                
                # 生産時間を計算
                if production > 0:
                    daily_setup_count += 1
                    setup_time = 30  # 段替え時間
                    production_time = (production / 込め数リスト[i]) * サイクルタイムリスト[i]
                    daily_production_time += production_time + setup_time
                
                inventory[i] += production
            
            # 在庫履歴に記録
            inventory_history[i].append(max(0, inventory[i]))
        
        # 稼働時間制約チェック（制約違反の場合は一部生産を削減）
        if daily_production_time > max_daily_work_time:
            # 簡易的な調整：超過分を比例配分で削減
            reduction_factor = max_daily_work_time / daily_production_time
            for i in range(品番数):
                if inventory[i] > 0:
                    # 生産量を削減し、在庫を調整
                    current_production = max(0, 出荷数リスト[i] - (initial_inventory[i] - inventory[i]))
                    reduced_production = current_production * reduction_factor
                    inventory[i] = initial_inventory[i] + reduced_production - 出荷数リスト[i] * (t + 1)
                    inventory[i] = max(0, inventory[i])
                    inventory_history[i][-1] = inventory[i]
    
    return inventory_history

def adjust_initial_inventory(初期在庫量リスト: List[float],
                           出荷数リスト: List[float],
                           サイクルタイムリスト: List[float],
                           込め数リスト: List[int],
                           在庫コスト単価: float = 180,
                           出荷遅れコスト単価: float = 500,
                           num_simulations: int = 50,
                           max_iterations: int = 3) -> List[float]:
    """初期在庫を調整する関数"""
    
    品番数 = len(初期在庫量リスト)
    s = 初期在庫量リスト[:]
    
    # h / (h+c) - 最適在庫理論に基づく計算
    prob_target = 在庫コスト単価 / (在庫コスト単価 + 出荷遅れコスト単価)
    
    print(f"\n=== 初期在庫水準の調整アルゴリズム開始 ===")
    
    for iteration in range(max_iterations):
        print(f"--- 調整イテレーション {iteration + 1} ---")
        
        # 各在庫点について在庫量の分布を求める
        inventory_distributions = [[] for _ in range(品番数)]
        for _ in range(num_simulations):
            inventory_history = simulate_production_schedule(s, 出荷数リスト, サイクルタイムリスト, 込め数リスト)
            for i in range(品番数):
                inventory_distributions[i].extend(inventory_history[i])
        
        adjustments = [0] * 品番数
        
        # 各在庫点について在庫量の最適調整量r^*を求める
        for i in range(品番数):
            if not inventory_distributions[i]:
                continue
            
            inventory_counts = pd.Series(inventory_distributions[i]).value_counts(normalize=True).sort_index()
            cumulative_distribution = inventory_counts.cumsum()
            
            # sum_{x <= r-1} f(x) <= h / (h + c) の条件を満たす最小のrを見つける
            best_r = 0
            for r in cumulative_distribution.index:
                prob_at_r_minus_1 = cumulative_distribution.get(r - 1, 0)
                if prob_at_r_minus_1 <= prob_target:
                    best_r = r
                else:
                    break
            
            # 調整量を計算
            current_avg = np.mean(inventory_distributions[i]) if inventory_distributions[i] else 0
            adjustments[i] = max(0, best_r - current_avg)
        
        print(f"  今回の調整量: {adjustments}")
        
        # 在庫量を更新
        for i in range(品番数):
            s[i] = max(0, s[i] - adjustments[i])
        
        print(f"  更新後の初期在庫量: {s}")
    
    print("--- 最大反復回数に到達しました。---")
    return s

def plot_inventory_comparison(品番リスト: List[str],
                            初期在庫量リスト: List[float],
                            調整後在庫量リスト: List[float],
                            file_name: str,
                            output_folder: str = 'initial_inventory',
                            font_size_base: int = 12) -> str:
    """調整前後の在庫量を比較する棒グラフを作成・保存する関数

    Args:
        品番リスト: 品番のリスト
        初期在庫量リスト: 調整前の在庫量リスト
        調整後在庫量リスト: 調整後の在庫量リスト
        file_name: ファイル名
        output_folder: 出力フォルダ
        font_size_base: 基本フォントサイズ（デフォルト: 12）
    """

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # プロットファイルのパス
    plot_path = os.path.join(output_folder, f'inventory_comparison_{file_name}.png')

    # 品番数が多い場合は上位20品番のみ表示
    max_items = 20
    if len(品番リスト) > max_items:
        # 調整前在庫量でソートして上位を選択
        sorted_indices = sorted(range(len(初期在庫量リスト)),
                              key=lambda i: 初期在庫量リスト[i], reverse=True)[:max_items]
        display_品番 = [品番リスト[i] for i in sorted_indices]
        display_調整前 = [初期在庫量リスト[i] for i in sorted_indices]
        display_調整後 = [調整後在庫量リスト[i] for i in sorted_indices]
        title_suffix = f" (上位{max_items}品番)"
    else:
        display_品番 = 品番リスト
        display_調整前 = 初期在庫量リスト
        display_調整後 = 調整後在庫量リスト
        title_suffix = ""

    # 図のサイズを設定
    fig_width = max(12, len(display_品番) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # x軸の位置
    x = np.arange(len(display_品番))
    width = 0.35

    # 棒グラフを作成
    bars1 = ax.bar(x - width/2, display_調整前, width, label='調整前', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, display_調整後, width, label='調整後', alpha=0.8, color='orange')

    # グラフの設定（文字サイズを調整可能に）
    ax.set_xlabel('品番', fontsize=font_size_base)
    ax.set_ylabel('在庫量', fontsize=font_size_base)
    ax.set_title(f'初期在庫量の調整前後比較 - {file_name}{title_suffix}', fontsize=font_size_base + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(display_品番, ha='right', fontsize=font_size_base - 2)
    ax.legend(fontsize=font_size_base)
    ax.grid(True, alpha=0.3)

    # y軸の目盛りラベルのフォントサイズも設定
    ax.tick_params(axis='y', labelsize=font_size_base - 1)

    # 右上に総在庫量の情報を追加
    total_before = sum(初期在庫量リスト)
    total_after = sum(調整後在庫量リスト)
    reduction = total_before - total_after
    reduction_rate = (reduction / total_before) * 100 if total_before > 0 else 0

    info_text = f'調整前総在庫: {total_before:,.0f}\n調整後総在庫: {total_after:,.0f}\n削減量: {reduction:,.0f}\n削減率: {reduction_rate:.1f}%'
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=font_size_base - 1)

    # 値をバーの上に表示
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(display_調整前) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=font_size_base - 4)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(display_調整前) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=font_size_base - 4)

    # レイアウトを調整
    plt.tight_layout()

    # 画像を保存
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"在庫比較プロットを保存: {plot_path}")
    return plot_path

def save_initial_inventory_to_csv(品番リスト: List[str],
                                初期在庫量リスト: List[float],
                                調整後在庫量リスト: List[float],
                                file_name: str,
                                output_folder: str = 'initial_inventory') -> str:
    """初期在庫をCSVファイルに保存する関数"""

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # CSVファイルのパス
    csv_path = os.path.join(output_folder, f'initial_inventory_{file_name}.csv')

    # データフレームを作成
    df = pd.DataFrame({
        '品番': 品番リスト,
        '調整前初期在庫': 初期在庫量リスト,
        '調整後初期在庫': 調整後在庫量リスト
    })

    # CSVファイルに保存
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"初期在庫CSVファイルを保存: {csv_path}")
    return csv_path

def process_single_file(csv_file: str, font_size: int = 12) -> Dict:
    """単一ファイルに対して初期在庫生成・調整を実行する関数"""
    file_name = os.path.basename(csv_file).replace('.csv', '')
    print(f"\n=== Processing {csv_file} ===")
    
    try:
        # データを読み込み
        品番リスト, 出荷数リスト, 収容数リスト, サイクルタイムリスト, 込め数リスト, 初期在庫量リスト = read_production_data(csv_file)
        
        print(f"品番数: {len(品番リスト)}")
        print(f"調整前初期在庫合計: {sum(初期在庫量リスト)}")
        
        # 初期在庫調整
        print("\n=== 初期在庫水準の調整アルゴリズム開始 ===")
        調整後在庫量リスト = adjust_initial_inventory(
            初期在庫量リスト, 出荷数リスト, サイクルタイムリスト, 込め数リスト
        )
        
        print(f"調整後初期在庫合計: {sum(調整後在庫量リスト)}")
        print(f"在庫削減量: {sum(初期在庫量リスト) - sum(調整後在庫量リスト)}")
        print(f"在庫削減率: {((sum(初期在庫量リスト) - sum(調整後在庫量リスト)) / sum(初期在庫量リスト)) * 100:.2f}%")
        
        # CSVファイルに保存
        csv_path = save_initial_inventory_to_csv(品番リスト, 初期在庫量リスト, 調整後在庫量リスト, file_name)

        # 在庫比較プロットを作成・保存
        plot_path = plot_inventory_comparison(品番リスト, 初期在庫量リスト, 調整後在庫量リスト, file_name, font_size_base=font_size)

        return {
            'file_name': file_name,
            'csv_path': csv_path,
            'plot_path': plot_path,
            '品番数': len(品番リスト),
            '調整前初期在庫合計': sum(初期在庫量リスト),
            '調整後初期在庫合計': sum(調整後在庫量リスト),
            '在庫削減量': sum(初期在庫量リスト) - sum(調整後在庫量リスト),
            '在庫削減率(%)': ((sum(初期在庫量リスト) - sum(調整後在庫量リスト)) / sum(初期在庫量リスト)) * 100 if sum(初期在庫量リスト) > 0 else 0
        }
        
    except Exception as e:
        print(f"エラーが発生しました ({csv_file}): {e}")
        return None

def process_all_files(data_folder: str = 'data', font_size: int = 12) -> List[Dict]:
    """データフォルダ内の全CSVファイルに対して初期在庫生成・調整を実行する関数"""
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    
    if not csv_files:
        print(f"データフォルダ '{data_folder}' にCSVファイルが見つかりません。")
        return []
    
    print(f"見つかったCSVファイル: {[os.path.basename(f) for f in csv_files]}")
    
    all_results = []
    
    for csv_file in csv_files:
        result = process_single_file(csv_file, font_size)
        if result:
            all_results.append(result)
    
    return all_results

def main(font_size: int = 12):
    """メイン実行関数"""
    print("=" * 60)
    print("初期在庫生成・調整モジュール")
    print("=" * 60)
    
    # データフォルダの確認
    data_folder = 'data'
    if not os.path.exists(data_folder):
        print(f"データフォルダ '{data_folder}' が見つかりません。")
        return
    
    # 全ファイル処理実行
    results = process_all_files(data_folder, font_size)
    
    if results:
        print(f"\n=== 処理完了 ===")
        print(f"処理ファイル数: {len(results)}")
        
        # 統計情報を表示
        total_items = sum(r['品番数'] for r in results)
        total_before = sum(r['調整前初期在庫合計'] for r in results)
        total_after = sum(r['調整後初期在庫合計'] for r in results)
        total_reduction = total_before - total_after
        
        print(f"\n=== 統計情報 ===")
        print(f"総品番数: {total_items}")
        print(f"総調整前在庫: {total_before:.0f}")
        print(f"総調整後在庫: {total_after:.0f}")
        print(f"総削減量: {total_reduction:.0f}")
        print(f"平均削減率: {(total_reduction / total_before) * 100:.2f}%")
        
        print(f"\n=== 生成されたファイル ===")
        for result in results:
            print(f"  CSV: {result['csv_path']}")
            print(f"  Plot: {result['plot_path']}")
    else:
        print("処理できるデータがありませんでした。")

if __name__ == "__main__":
    # デフォルトのフォントサイズ（12）で実行
    main(font_size=20)
