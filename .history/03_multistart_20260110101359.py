"""
多スタートアニーリング法による生産スケジューリング
"""

import os
import time
import pandas as pd
import process

# ==========================================
# 設定エリア
# ==========================================
# 処理したいファイル名（拡張子なし）
TARGET_FILE = 'D40'

# 最適化パラメータ
NUM_STARTS = 30
MAX_ITERATIONS = 5000
INITIAL_TEMP = 5000
COOLING_RATE = 0.999
DATA_FOLDER = 'data'

def main():
    # パス生成
    main_csv = os.path.join(DATA_FOLDER, f'{TARGET_FILE}.csv')
    ship_csv = os.path.join(DATA_FOLDER, f'{TARGET_FILE}_ship.csv')
    
    # ---------------------------------------------------------
    # 1. ファイルチェックとデータ読み込み
    # ---------------------------------------------------------
    if not os.path.exists(main_csv):
        print(f"メインCSVが見つかりません: {main_csv}")
        return
    if not os.path.exists(ship_csv):
        print(f"出荷数CSVが見つかりません: {ship_csv}")

    process.read_csv(main_csv, ship_csv)
    print(f"データ読込み完了")

    # ---------------------------------------------------------
    # 2. 出荷数データの可視化
    # ---------------------------------------------------------
    if process.出荷数マトリクス:
        print("出荷数データをグラフ化しています...")
        process.visualize_demand_data(process.出荷数マトリクス, process.品番リスト, TARGET_FILE)

    # ---------------------------------------------------------
    # 3. 初期在庫の読み込み
    # ---------------------------------------------------------
    print("初期在庫の読み込みをしています...")
    try:
        adjusted_inventory = process.load_initial_inventory_from_csv(TARGET_FILE)
    except:
        print("初期在庫ファイルが見つからないので処理を中止します")
        return

    # ---------------------------------------------------------
    # 4. 多スタートアニーリング法の実行
    # ---------------------------------------------------------
    print(f"最適化計算を開始します... (開始点: {NUM_STARTS}, 最大反復回数: {MAX_ITERATIONS}) ")
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
    print("計算完了")
    print(f"最良コスト: {best_cost:,.2f}")
    print(f"計算時間　: {calc_time:.2f}秒")

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
    print(f"結果をCSVに保存しました: {csv_filename}")

    # 詳細結果プロットとCSV出力
    violation_count = process.plot_result(best_solution, adjusted_inventory, TARGET_FILE, "SA", 期間=20)
    
    if violation_count > 0:
        print(f"【注意】時間制約違反が発生している日があります: {violation_count}日")

if __name__ == "__main__":
    main()