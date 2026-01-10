#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合整数計画法(MIP)による生産スケジューリング実行スクリプト
（単一ファイル指定実行版）
"""

import os
import time
import pandas as pd
import process  # process.py に必要な関数が集約されている前提

# ==========================================
# 設定エリア
# ==========================================
# ★ここに処理したいファイル名を直接指定（拡張子なし）
TARGET_FILE = 'D40'

# MIPパラメータ
TIME_LIMIT = 500        # 計算時間制限（秒）
DATA_FOLDER = 'data'    # データフォルダ

def main():
    print("=" * 60)
    print(f"MIP(混合整数計画法) スケジューリング実行: 対象ファイル [{TARGET_FILE}]")
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
    # 4. 最適化実行 (MIP)
    # ---------------------------------------------------------
    print(f"\n=== 最適化計算開始 (TimeLimit={TIME_LIMIT}s) ===")
    start_time = time.time()
    
    # MIPソルバーを実行
    # solve_mip は (schedule, cost, info_dict) を返す想定
    production_schedule, total_cost, accuracy_info = process.solve_mip(
        adjusted_inventory, 
        time_limit=TIME_LIMIT
    )
    
    calc_time = time.time() - start_time

    # ---------------------------------------------------------
    # 5. 結果表示と保存
    # ---------------------------------------------------------
    if production_schedule is None:
        print("\n【失敗】解が見つかりませんでした（またはタイムアウトで解なし）。")
        return

    print(f"\n=== 計算完了 ===")
    print(f"総コスト　: {total_cost:,.2f}")
    print(f"計算時間　: {calc_time:.2f} 秒")
    if accuracy_info:
        print(f"ソルバー情報: {accuracy_info}")

    # CSV保存用データの作成
    result_data = {
        'file_name': TARGET_FILE,
        'method': 'MIP',
        'total_cost': total_cost,
        'calc_time': calc_time,
        'time_limit': TIME_LIMIT
    }
    # MIP特有の情報があれば追加（Gapなど）
    if accuracy_info:
        result_data.update(accuracy_info)

    csv_filename = f"MIP_results_{TARGET_FILE}.csv"
    process.save_results_to_csv([result_data], csv_filename)
    print(f"結果CSV保存: result/{csv_filename}")

    # 詳細結果プロットとCSV出力
    violation_count = process.plot_result(
        production_schedule, 
        adjusted_inventory, 
        TARGET_FILE, 
        method_name="MIP", 
        期間=20
    )
    
    if violation_count > 0:
        print(f"【注意】時間制約違反が発生している日があります: {violation_count}日")

if __name__ == "__main__":
    main()