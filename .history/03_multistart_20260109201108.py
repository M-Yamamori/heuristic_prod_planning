#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適化されたMultistart焼きなまし法による生産スケジューリング
"""

import os
import time
import pandas as pd
import process  # visualize_demand_data は process.py 内にある前提

# ==========================================
# 設定エリア
# ==========================================
# ★ここに処理したいファイル名を直接指定（拡張子なし）
TARGET_FILE = 'D40'

# 最適化パラメータ
NUM_STARTS = 30         # スタート数
MAX_ITERATIONS = 5000  # 最大反復回数
INITIAL_TEMP = 5000    # 初期温度
COOLING_RATE = 0.999      # 冷却率
DATA_FOLDER = 'data'    # データフォルダ

def main():
    print("=" * 60)
    print(f"生産スケジューリング実行: 対象ファイル [{TARGET_FILE}]")
    print("=" * 60)

    # パス生成
    main_csv = os.path.join(DATA_FOLDER, f'{TARGET_FILE}.csv')
    ship_csv = os.path.join(DATA_FOLDER, f'{TARGET_FILE}_ship.csv')
    
    # ---------------------------------------------------------
    # 1. ファイルチェックとデータ読み込み
    # ---------------------------------------------------------
    if not os.path.exists(main_csv):
        print(f"【エラー】メインCSVが見つかりません: {main_csv}")
        return
    if not os.path.exists(ship_csv):
        print(f"【警告】需要詳細CSVが見つかりません: {ship_csv}")

    # データを process モジュールのグローバル変数にロード
    process.read_csv(main_csv, ship_csv)

    print(f"データ読込完了: 品番数 {len(process.品番リスト)}")

    # ---------------------------------------------------------
    # 2. 需要データの可視化 (process.pyに追加した関数を使用)
    # ---------------------------------------------------------
    if process.出荷数マトリクス:
        print("需要データをグラフ化しています...")
        process.visualize_demand_data(process.出荷数マトリクス, process.品番リスト, TARGET_FILE)

    # ---------------------------------------------------------
    # 3. 初期在庫の読み込み
    # ---------------------------------------------------------
    print(f"\n=== 初期在庫の読み込み ===")
    try:
        adjusted_inventory = process.load_initial_inventory_from_csv(TARGET_FILE)
    except Exception:
        print("初期在庫ファイルが見つからないため、在庫0から開始します。")
        adjusted_inventory = [0] * len(process.品番リスト)

    # ---------------------------------------------------------
    # 4. 最適化実行 (Multi-start SA)
    # ---------------------------------------------------------
    print(f"\n=== 最適化計算開始 (Starts={NUM_STARTS}, Iter={MAX_ITERATIONS}) ===")
    start_time = time.time()
    
    best_solution, best_cost = process.multi_start_simulated_annealing(
        adjusted_inventory,
        num_starts=NUM_STARTS,
        max_iterations=MAX_ITERATIONS,
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE
    )
    
    calc_time = time.time() - start_time

    # ---------------------------------------------------------
    # 5. 結果表示と保存
    # ---------------------------------------------------------
    print(f"\n=== 計算完了 ===")
    print(f"最良コスト: {best_cost:,.2f}")
    print(f"計算時間　: {calc_time:.2f} 秒")

    # CSV保存
    csv_filename = f"SA_results_{TARGET_FILE}.csv"
    result_data = [{
        'file_name': TARGET_FILE,
        'total_cost': best_cost,
        'calc_time': calc_time,
        'num_starts': NUM_STARTS,
        'max_iterations': MAX_ITERATIONS
    }]
    process.save_results_to_csv(result_data, csv_filename)
    print(f"結果CSV保存: result/{csv_filename}")

    # 詳細結果プロットとCSV出力
    violation_count = process.plot_result(best_solution, adjusted_inventory, TARGET_FILE, "SA", 期間=20)
    
    if violation_count > 0:
        print(f"【注意】時間制約違反が発生している日があります: {violation_count}日")

if __name__ == "__main__":
    main()