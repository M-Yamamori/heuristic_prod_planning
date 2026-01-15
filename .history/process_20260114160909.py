import pandas as pd
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import glob
import time
import sys
import math

# ==========================================
# グローバル変数・設定
# ==========================================
品番リスト = []
出荷数リスト = []
収容数リスト = []
サイクルタイムリスト = []
込め数リスト = []
初期在庫量リスト = []
出荷数マトリクス = []

# コストパラメータ
在庫コスト単価 = 200
残業コスト単価 = 60
段替えコスト単価 = 600
出荷遅れコスト単価 = 50000

# 時間設定（分）
定時 = 8 * 60 * 2  # 960分
最大残業時間 = 2 * 60 * 2  # 240分
段替え時間 = 30

# グラフY軸設定
PLOT_YLIM_WORKING = (0, 1300)
PLOT_YLIM_INVENTORY = (0, 4000)
PLOT_YLIM_PRODUCTION = (0, 4000)
PLOT_YLIM_SHORTAGE = (0, 300)
PLOT_YLIM_DEMAND = (0, 3500)


# ==========================================
# データ取得・ヘルパー関数
# ==========================================
def get_demand(item_index, period_index, 期間=20):
    if 出荷数マトリクス:
        if 0 <= item_index < len(出荷数マトリクス):
            row = 出荷数マトリクス[item_index]
            if 0 <= period_index < len(row):
                return row[period_index]
        return 0
    else:
        return 出荷数リスト[item_index] if item_index < len(出荷数リスト) else 0

def get_avg_demand():
    if 出荷数マトリクス:
        vals = [v for row in 出荷数マトリクス for v in row]
        return sum(vals) / len(vals) if vals else 0
    else:
        return sum(出荷数リスト) / len(出荷数リスト) if 出荷数リスト else 0

def get_avg_demand_for_item(i):
    if 出荷数マトリクス:
        if 0 <= i < len(出荷数マトリクス):
            row = 出荷数マトリクス[i]
            return sum(row) / len(row) if row else 0
        return 0
    else:
        return 出荷数リスト[i] if i < len(出荷数リスト) else 0

def visualize_demand_data(shipment_matrix, item_list, file_name):
    if not shipment_matrix or len(shipment_matrix) == 0:
        print("出荷量データがありません")
        return

    期間 = len(shipment_matrix[0])
    cols = [f'{t+1}日' for t in range(期間)]
    df = pd.DataFrame(shipment_matrix, index=item_list, columns=cols)

    os.makedirs('result', exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))

    demand_data = np.array(shipment_matrix)
    colors = plt.cm.Set3(np.linspace(0, 1, len(item_list)))
    bot = np.zeros(期間)
    for i in range(min(10, len(item_list))):
        ax.bar(range(1, 期間+1), demand_data[i], bottom=bot, label=str(i+1), color=colors[i], alpha=0.8)
        bot += demand_data[i]
    if len(item_list) > 10:
        ax.bar(range(1, 期間+1), np.sum(demand_data[10:], axis=0), bottom=bot, label='Others', color='gray', alpha=0.8)
    ax.set_title('出荷量', fontsize=20, fontweight='bold')
    ax.set_xlabel('日', fontsize=18)
    ax.set_ylabel('個数', fontsize=18)
    ax.set_xticks(range(1, 期間+1))
    ax.set_xticklabels([str(i) for i in range(1, 期間+1)], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    if PLOT_YLIM_DEMAND:
        ax.set_ylim(PLOT_YLIM_DEMAND)

    plt.tight_layout()
    save_path = f'result/demand_visualization_{file_name}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"出荷数データのグラフを保存しました: {save_path}")

def classify_item(i):
    """
    品番をHigh/Mid/Lowに分類する
    """
    avg_i = get_avg_demand_for_item(i)
    avg_all = get_avg_demand()
    
    if avg_i > avg_all * 1.5:
        return 'High'
    elif avg_i < avg_all * 0.5:
        return 'Low'
    else:
        return 'Mid'

def calculate_item_inventory_profile(item_idx, production_row, initial_inventory_val, 期間):
    inventory = []
    current = initial_inventory_val
    for t in range(期間):
        current += production_row[t]
        current -= get_demand(item_idx, t)
        inventory.append(current)
    return inventory


# ==========================================
# ファイル読み込み
# ==========================================

def read_csv(main_file_path, ship_file_path):
    global 品番リスト, 出荷数リスト, 収容数リスト, サイクルタイムリスト, 込め数リスト, 初期在庫量リスト, 出荷数マトリクス
    
    if not os.path.exists(main_file_path):
        print(f"メインファイルが見つかりません: {main_file_path}")
        sys.exit(1)

    base_dir = os.path.dirname(main_file_path)
    if base_dir == '': base_dir = '.'
    
    収容数辞書 = {}
    capacity_path = os.path.join(base_dir, '収容数.csv')
    if not os.path.exists(capacity_path):
        capacity_path = '収容数.csv' 

    if os.path.exists(capacity_path):
        try:
            with open(capacity_path, 'r', encoding='shift-jis') as f:
                r = csv.reader(f)
                next(r, None)
                for row in r:
                    if len(row) >= 2 and row[1].strip():
                        try: 収容数辞書[str(row[0]).strip()] = int(float(row[1]))
                        except: pass
        except Exception as e:
            print(f"収容数ファイルが見つかりません: {e}")

    print(f"メインデータ読み込み: {main_file_path}")
    
    品番リスト = []
    出荷数リスト = []
    収容数リスト = []
    サイクルタイムリスト = []
    込め数リスト = []
    期間数 = 20

    try:
        f = open(main_file_path, 'r', encoding='shift-jis')
        reader = csv.reader(f)
        header = next(reader)
        
        with f:
            header = [str(h).strip() for h in header]
            try:
                idx_qty = header.index("個数")
                idx_code = header.index("素材品番")
                idx_kome = header.index("込数")
                idx_ct = header.index("サイクルタイム")
            except ValueError as e:
                print(f"メインファイルに必要な列名が見つかりません: {header}")
                sys.exit(1)

            for row in reader:
                if not row: continue
                try:
                    total_qty = int(float(row[idx_qty]))
                    if total_qty < 200: continue
                    
                    p_code = str(row[idx_code]).strip()
                    品番リスト.append(p_code)
                    出荷数リスト.append(int(total_qty / 期間数)) 
                    込め数リスト.append(int(float(row[idx_kome])))
                    ct_val = row[idx_ct]
                    ct_float = float(ct_val) if ct_val.strip() else 0.0
                    サイクルタイムリスト.append(ct_float / 60)
                    収容数リスト.append(収容数辞書.get(p_code, 80))
                except ValueError:
                    continue

    except Exception as e:
        print(f"メインCSVの読込みエラー: {e}")
        sys.exit(1)

    出荷数マトリクス = []
    if not ship_file_path or not os.path.exists(ship_file_path):
        print(f"出荷数ファイルが見つかりません: {ship_file_path}")
        sys.exit(1)
        
    ship_data = {}
    try:
        df = pd.read_csv(ship_file_path, encoding='shift-jis')
        
        df.columns = [str(c).strip() for c in df.columns]
        code_col = None
        if '素材品番' in df.columns: code_col = '素材品番'
        elif '品番' in df.columns: code_col = '品番'
        else: code_col = df.columns[0]
        
        df[code_col] = df[code_col].astype(str).str.strip()
        cols = [c for c in df.columns if str(c).isdigit() and 1 <= int(c) <= 期間数]
        
        for _, r in df.iterrows():
            vals = pd.to_numeric(r[cols], errors='coerce').fillna(0).astype(int).values.tolist()
            ship_data[r[code_col]] = vals
            
    except Exception as e:
        print(f"出荷数データの読込みエラー: {e}")
        sys.exit(1)

    match_count = 0
    missing_codes = []
    for i, code in enumerate(品番リスト):
        clean_code = str(code).strip()
        vals = ship_data.get(clean_code)
        if vals is None:
            vals = [0] * 期間数
            missing_codes.append(clean_code)
        else:
            match_count += 1
        if len(vals) < 期間数: vals += [0] * (期間数 - len(vals))
        else: vals = vals[:期間数]
        出荷数マトリクス.append(vals)
        if 期間数 > 0:
            出荷数リスト[i] = int(sum(vals) / 期間数)
    
    print(f"出荷数データの読み込み完了（{len(品番リスト)}件中 {match_count}件 ヒット）")
    if missing_codes:
        print(f"未ヒット品番: {missing_codes[:5]}")

    return 品番リスト, 出荷数リスト, 収容数リスト, サイクルタイムリスト, 込め数リスト

def load_initial_inventory_from_csv(file_name, inventory_folder='initial_inventory'):
    path = os.path.join(inventory_folder, f'initial_inventory_{file_name}.csv')
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df['調整後初期在庫'].fillna(0).astype(int).tolist()


# ==========================================
# 近傍操作関数群
# ==========================================

def check_jit_rule(item_idx, production_row, initial_inv_val, 期間):
    """
    「在庫のある品番は生産しない」をチェックする関数
    """
    current_inv = initial_inv_val
    for t in range(期間):
        dem = get_demand(item_idx, t)
        
        # 「在庫 >= 需要」なのに生産するのはダメ
        prod = production_row[t]
        if prod > 0:
            if current_inv >= dem:
                return False
        
        current_inv += prod - dem
    return True

def _op_move_to_previous_day(new_solution, initial_inventory, 期間):
    """
    残業が発生している日の生産を余裕がある前日に移動する関数
    （隣接した日のみ移動可能）
    """
    daily_loads = _compute_daily_loads(new_solution, 期間)
    
    # 負荷が高い順に日を探す（初日は前日がないので除外）
    busy_days_indices = np.argsort(daily_loads)[::-1]
    
    target_busy_day = -1
    for day in busy_days_indices:
        if day > 0 and daily_loads[day] > 定時: # 定時を超えている日を優先
            target_busy_day = day
            break
            
    if target_busy_day == -1:
        # 残業がなくても定時ギリギリなら分散を試みる
        for day in busy_days_indices:
            if day > 0 and daily_loads[day] > daily_loads[day-1]:
                target_busy_day = day
                break
    
    if target_busy_day == -1: return new_solution

    prev_day = target_busy_day - 1
    
    # 前日がすでにパンパンならやめる
    if daily_loads[prev_day] >= 定時 * 0.95: return new_solution

    # その日に生産されている品番を取得
    品番数 = len(initial_inventory)
    candidates = [i for i in range(品番数) if new_solution[i][target_busy_day] > 0]
    random.shuffle(candidates)

    original_sol = [row[:] for row in new_solution]

    for i in candidates:
        amount = new_solution[i][target_busy_day]
        
        # 移動量の決定（全量移動/半分移動）
        move_qty = amount
        if random.random() < 0.5:
            move_qty = int(amount * 0.5)
        
        if move_qty <= 0: continue

        # 移動試行
        new_solution[i][target_busy_day] -= move_qty
        new_solution[i][prev_day] += move_qty

        # 1.JITルールチェック
        if not check_jit_rule(i, new_solution[i], initial_inventory[i], 期間):
            # 在庫過多なら元に戻して次へ
            new_solution[i][target_busy_day] += move_qty
            new_solution[i][prev_day] -= move_qty
            continue
        
        # 2.前日の負荷チェック（前日が残業にならないように）
        added_load = (move_qty / 込め数リスト[i]) * サイクルタイムリスト[i]
        # 段替えが増える場合は段替え時間を加算
        if original_sol[i][prev_day] == 0:
            added_load += 段替え時間
            
        if daily_loads[prev_day] + added_load > 定時 * 1.05:
             # 負荷方なら元に戻して次へ
            new_solution[i][target_busy_day] += move_qty
            new_solution[i][prev_day] -= move_qty
            continue

        return new_solution

    return original_sol

def _op_fill_valleys(new_solution, initial_inventory, 期間):
    """
    最も稼働が低い日に最も稼働が高い日から負荷を追加する関数
    （離れた日も移動可能）
    """
    daily_loads = _compute_daily_loads(new_solution, 期間)
    min_day = int(np.argmin(daily_loads))
    max_day = int(np.argmax(daily_loads))

    if min_day == max_day: return new_solution
    if daily_loads[max_day] - daily_loads[min_day] < 定時 * 0.2: return new_solution # 差が小さければやらない

    # max_dayの品番をmin_dayへ移動
    品番数 = len(initial_inventory)
    candidates = [i for i in range(品番数) if new_solution[i][max_day] > 0]
    random.shuffle(candidates)

    original_sol = [row[:] for row in new_solution]

    for i in candidates:
        amount = new_solution[i][max_day]
        
        # 全量移動を優先（30%の確率で分割も許容）
        move_qty = amount
        if amount > 10 and random.random() < 0.3:
            move_qty = int(amount / 2)
            
        # 仮移動
        new_solution[i][max_day] -= move_qty
        new_solution[i][min_day] += move_qty
        
        # チェック
        # 1.欠品
        if min(calculate_item_inventory_profile(i, new_solution[i], initial_inventory[i], 期間)) < 0:
            new_solution[i] = list(original_sol[i])
            continue
        
        # 2.JIT
        if not check_jit_rule(i, new_solution[i], initial_inventory[i], 期間):
            new_solution[i] = list(original_sol[i])
            continue

        return new_solution

    return original_sol

def _op_eliminate_shortage(new_solution, initial_inventory, 期間):
    """
    欠品を解消するために生産を追加・移動する関数
    """
    品番数 = len(initial_inventory)
    candidates = list(range(品番数))
    random.shuffle(candidates)
    
    target_item = -1; target_day = -1
    for i in candidates:
        inv_profile = calculate_item_inventory_profile(i, new_solution[i], initial_inventory[i], 期間)
        if min(inv_profile) < 0:
            target_item = i
            for t in range(期間):
                if inv_profile[t] < 0:
                    target_day = t; break
            break
            
    if target_item == -1: return new_solution 
    
    needed = abs(calculate_item_inventory_profile(target_item, new_solution[target_item], initial_inventory[target_item], 期間)[target_day])
    prod_days = [t for t in range(target_day + 1) if new_solution[target_item][t] > 0]
    add_day = prod_days[-1] if prod_days else target_day
    
    new_solution[target_item][add_day] += int(needed) + 1
    return new_solution

def _strat_high_volume_jit(new_solution, i, initial_inventory, 期間):
    """
    【大量品】毎日必要な分だけ作る補助関数
    """
    current_inv = initial_inventory[i]
    for t in range(期間):
        dem = get_demand(i, t)
        if current_inv >= dem:
            current_inv -= dem
            new_solution[i][t] = 0
        else:
            prod = dem - current_inv
            new_solution[i][t] = int(prod)
            current_inv = 0
    return new_solution

def _strat_low_mid_batch(new_solution, i, initial_inventory, 期間, batch_size=3):
    """
    【中・少量品】在庫切れ時にまとめて生産する補助関数
    """
    current_inv = initial_inventory[i]
    new_solution[i] = [0] * 期間
    
    t = 0
    while t < 期間:
        dem = get_demand(i, t)
        if current_inv >= dem:
            current_inv -= dem
            t += 1
        else:
            needed = 0
            for k in range(batch_size):
                if t + k < 期間:
                    needed += get_demand(i, t + k)
            
            prod = max(0, needed - current_inv)
            new_solution[i][t] = int(prod)
            current_inv += prod - dem
            t += 1
            
    return new_solution

def _op_optimize_item_pattern(new_solution, initial_inventory, 期間):
    """
    出荷量に応じた品番の生産パターンを適用する関数
    """
    品番数 = len(initial_inventory)
    i = random.randint(0, 品番数 - 1)
    itype = classify_item(i)
    
    # High: daily JIT
    if itype == 'High':
        return _strat_high_volume_jit(new_solution, i, initial_inventory, 期間)
    # Mid: occasionally prefer JIT, otherwise smaller batch sizes
    elif itype == 'Mid':
        b_size = random.choice([2, 3, 4])
        return _strat_low_mid_batch(new_solution, i, initial_inventory, 期間, batch_size=b_size)
    # Low: smaller batches but occasionally slightly larger
    else: # Low
        b_size = random.choice([4, 5, 6])
        return _strat_low_mid_batch(new_solution, i, initial_inventory, 期間, batch_size=b_size) 

def _op_push_production_forward(new_solution, initial_inventory, 期間):
    """
    生産を後ろ倒しして在庫を減らす関数
    """
    original_sol = [row[:] for row in new_solution]
    品番数 = len(initial_inventory)
    
    for _ in range(10):
        i = random.randint(0, 品番数 - 1)
        prod_days = [t for t in range(期間) if new_solution[i][t] > 0]
        if prod_days: break
    else:
        return new_solution
        
    src_day = random.choice(prod_days)
    amount = new_solution[i][src_day]
    
    candidates = [t for t in range(src_day + 1, min(src_day + 7, 期間))]
    if not candidates: return new_solution
    candidates.reverse() 
    
    for dst_day in candidates:
        new_solution[i][src_day] = 0
        new_solution[i][dst_day] += amount
        
        # チェック
        # 1.欠品
        inv_profile = calculate_item_inventory_profile(i, new_solution[i], initial_inventory[i], 期間)
        if min(inv_profile) < 0:
            new_solution[i][src_day] = amount
            new_solution[i][dst_day] -= amount
            continue

        # 2.JIT
        if not check_jit_rule(i, new_solution[i], initial_inventory[i], 期間):
            new_solution[i][src_day] = amount
            new_solution[i][dst_day] -= amount
            continue
            
        return new_solution 
        
    return original_sol

def _op_reduce_setups(new_solution, initial_inventory, 期間):
    """
    段替えを減らすために隣接日の生産を統合する関数
    """
    strategies = [(_strat_adjacent_merge, 4), (_strat_move_day, 3)]
    func = random.choices([s[0] for s in strategies], weights=[s[1] for s in strategies])[0]
    return func(new_solution, initial_inventory, 期間)

def _strat_adjacent_merge(new_solution, initial_inventory, 期間):
    """
    隣接日の生産を統合する補助関数
    """
    i = random.randint(0, len(new_solution)-1)
    if classify_item(i) == 'High': return new_solution
    for t in range(期間-1):
        if new_solution[i][t] > 0 and new_solution[i][t+1] > 0:
            total = new_solution[i][t] + new_solution[i][t+1]
            if random.random() < 0.5:
                new_solution[i][t] = total; new_solution[i][t+1] = 0
            else:
                inv_prof = calculate_item_inventory_profile(i, new_solution[i], initial_inventory[i], 期間)
                if inv_prof[t] - new_solution[i][t] >= 0:
                    new_solution[i][t] = 0; new_solution[i][t+1] = total
            return new_solution
    return new_solution

def _strat_move_day(new_solution, initial_inventory, 期間):
    """
    隣接日に生産を移動する補助関数
    """
    i = random.randint(0, len(new_solution)-1)
    if classify_item(i) == 'High': return new_solution
    days = [t for t in range(期間) if new_solution[i][t] > 0]
    if days:
        src = random.choice(days)
        dst = src + random.choice([-1, 1])
        if 0 <= dst < 期間:
            can_move = True
            if src < dst:
                inv_prof = calculate_item_inventory_profile(i, new_solution[i], initial_inventory[i], 期間)
                if inv_prof[src] < new_solution[i][src]: can_move = False
            if can_move:
                new_solution[i][dst] += new_solution[i][src]
                new_solution[i][src] = 0
    return new_solution

def _compute_daily_loads(sol, 期間):
    """
    指定日の作業負荷を計算して返す補助関数
    """
    品番数 = len(sol)
    loads = [0] * 期間
    for t in range(期間):
        for i in range(品番数):
            if sol[i][t] > 0:
                loads[t] += 段替え時間 + (sol[i][t] / 込め数リスト[i]) * サイクルタイムリスト[i]
    return loads

def _op_balance_workload(new_solution, initial_inventory, 期間):
    """
    負荷の分割移動・スワップ・複数品番同時移動を試みる関数
    """
    original_sol = [row[:] for row in new_solution]
    品番数 = len(initial_inventory)

    daily_loads = _compute_daily_loads(new_solution, 期間)
    busy_day = int(np.argmax(daily_loads))

    # まだ過負荷でない場合は確率でスキップ
    if daily_loads[busy_day] <= 定時 * 1.02 and random.random() < 0.85:
        return new_solution

    # 移動先を探す
    lazy_days = [t for t in range(期間) if daily_loads[t] < 定時 * 0.9]
    if not lazy_days:
        return new_solution

    # 移動候補品番を探す
    candidates = [i for i in range(品番数) if new_solution[i][busy_day] > 0 and classify_item(i) != 'High']
    random.shuffle(candidates)

    for i in candidates:
        amount = new_solution[i][busy_day]
        if amount <= 0: continue

        # 分割して複数日に振り分ける試行
        for pct in [0.5, 0.4, 0.3, 0.2, 1.0]:
            moved = max(1, int(amount * pct))
            targets = sorted(lazy_days, key=lambda d: (abs(d - busy_day), daily_loads[d]))[:3]
            remaining = moved
            tmp_sol = [row[:] for row in new_solution]
            tmp_loads = daily_loads[:]

            for idx, dst in enumerate(targets):
                if remaining <= 0: break
                # 移動量決定
                portion = remaining if idx == 0 else int(moved / len(targets))
                portion = max(1, min(portion, remaining))

                added_time = 段替え時間 + (portion / 込め数リスト[i]) * サイクルタイムリスト[i]
                # 宛先で過負荷にならないか確認
                if tmp_loads[dst] + added_time > 定時 * 1.05:
                    continue

                tmp_sol[i][busy_day] -= portion
                tmp_sol[i][dst] += portion
                tmp_loads[busy_day] -= (段替え時間 + (portion / 込め数リスト[i]) * サイクルタイムリスト[i])
                tmp_loads[dst] += added_time
                remaining -= portion

            if remaining > 0:
                continue

            # チェック
            # 欠品・JIT
            if min(calculate_item_inventory_profile(i, tmp_sol[i], initial_inventory[i], 期間)) < 0:
                continue
            if not check_jit_rule(i, tmp_sol[i], initial_inventory[i], 期間):
                continue

            # ピークが減っていなければ元に戻す
            if max(tmp_loads) >= max(daily_loads) - 1e-6:
                continue

            return tmp_sol

    # スワップ操作
    busy_candidates = [i for i in range(品番数) if new_solution[i][busy_day] > 0 and classify_item(i) != 'High']
    for dst in [t for t in range(期間) if daily_loads[t] < 定時 * 0.95]:
        dst_candidates = [j for j in range(品番数) if new_solution[j][dst] > 0 and classify_item(j) != 'High']
        for i in busy_candidates:
            for j in dst_candidates:
                amt = min(new_solution[i][busy_day], new_solution[j][dst])
                if amt <= 0: continue
                tmp_sol = [row[:] for row in new_solution]
                tmp_sol[i][busy_day] -= amt; tmp_sol[i][dst] += amt
                tmp_sol[j][dst] -= amt; tmp_sol[j][busy_day] += amt

                loads = _compute_daily_loads(tmp_sol, 期間)
                # 交換によって最大負荷が下がって移動先が許容内なら受け入れ
                if loads[busy_day] <= daily_loads[busy_day] - 1 and loads[dst] <= 定時 * 1.05:
                    if min(calculate_item_inventory_profile(i, tmp_sol[i], initial_inventory[i], 期間)) < 0: continue
                    if min(calculate_item_inventory_profile(j, tmp_sol[j], initial_inventory[j], 期間)) < 0: continue
                    if not check_jit_rule(i, tmp_sol[i], initial_inventory[i], 期間): continue
                    if not check_jit_rule(j, tmp_sol[j], initial_inventory[j], 期間): continue
                    return tmp_sol

    return original_sol

def _op_aggressive_gather(new_solution, initial_inventory, 期間):
    """
    ある品番の生産を既存の生産日のうちの1日に全集約する関数
    """
    品番数 = len(initial_inventory)
    
    # 生産日が2日以上ある品番を探す
    candidates_items = []
    for i in range(品番数):
        # 生産がある日のリスト
        prod_days = [t for t in range(期間) if new_solution[i][t] > 0]
        if len(prod_days) >= 2:
            candidates_items.append(i)
            
    if not candidates_items:
        return new_solution

    # ランダムに品番を選択
    i = random.choice(candidates_items)
    prod_days = [t for t in range(期間) if new_solution[i][t] > 0]
    
    # 集約先の日を決める（ランダム）
    target_day = random.choice(prod_days)
    
    # 集約
    original_row = new_solution[i][:]
    
    total_amount = sum(original_row)
    new_solution[i] = [0] * 期間
    new_solution[i][target_day] = total_amount # ターゲット日に全量入れる

    # 在庫コストが大きく増えないか確認
    def _item_inv_cost(row):
        inv = initial_inventory[i]
        cost = 0.0
        for t in range(期間):
            inv += row[t] - get_demand(i, t)
            if inv > 0:
                cost += (inv / 収容数リスト[i]) * 在庫コスト単価
            elif inv < 0:
                inv = 0
        return cost

    before_cost = _item_inv_cost(original_row)
    after_cost = _item_inv_cost(new_solution[i])

    # 在庫コストが10%を超えて増える場合は元に戻す
    if after_cost > before_cost * 1.10:
        new_solution[i] = original_row
        return new_solution

    # チェック
    # JIT
    if not check_jit_rule(i, new_solution[i], initial_inventory[i], 期間):
        new_solution[i] = original_row
        return new_solution

    return new_solution

def generate_neighbor(solution, initial_inventory, 期間=20):
    new_solution = [row[:] for row in solution]
    
    operations = [
        _op_eliminate_shortage,
        _op_optimize_item_pattern,
        _op_balance_workload,
        _op_move_to_previous_day,
        _op_fill_valleys,
        _op_reduce_setups,
        _op_aggressive_gather,
        _op_push_production_forward
    ]
    
    # 重み
    weights = [
        10, # eliminate_shortage
        5,  # optimize_item_pattern
        30, # balance_workload
        20, # move_to_previous_day
        20, # fill_valleys
        50, # reduce_setups
        100, # aggressive_gather
        5   # push_production_forward
    ]
    
    chosen_op = random.choices(operations, weights=weights, k=1)[0]
    return chosen_op(new_solution, initial_inventory, 期間)

# ==========================================
# SAメインロジック
# ==========================================

def calculate_total_cost(production_schedule, initial_inventory, 期間=20):
    品番数 = len(initial_inventory)
    inv_cost = 0; setup_cost = 0; ot_cost = 0; shortage_cost = 0
    current_inv = list(initial_inventory)
    
    for t in range(期間):
        daily_time = 0
        for i in range(品番数):
            prod = production_schedule[i][t]
            if prod > 0:
                setup_cost += 段替えコスト単価
                daily_time += 段替え時間 + (prod / 込め数リスト[i]) * サイクルタイムリスト[i]
            current_inv[i] += prod - get_demand(i, t)
            
            if current_inv[i] > 0:
                inv_cost += (current_inv[i] / 収容数リスト[i]) * 在庫コスト単価
            elif current_inv[i] < 0:
                shortage_cost += (abs(current_inv[i]) / 収容数リスト[i]) * 出荷遅れコスト単価 * 10
                current_inv[i] = 0
                
        if daily_time > 定時:
            over = daily_time - 定時
            if over <= 最大残業時間:
                ot_cost += over * 残業コスト単価
            else:
                ot_cost += 最大残業時間 * 残業コスト単価
                ot_cost += (over - 最大残業時間) * 残業コスト単価 * 10000 
                
    return inv_cost + setup_cost + ot_cost + shortage_cost

def simulated_annealing_scheduler(initial_inventory, max_iterations=300000, initial_temp=5000, cooling_rate=0.999):
    """SAスケジューラ"""
    品番数 = len(initial_inventory)
    期間 = 20
    if 品番数 == 0: return None, None

    # 初期解生成
    current_sol = [[0]*期間 for _ in range(品番数)]
    for i in range(品番数):
        current_sol = _op_optimize_item_pattern(current_sol, initial_inventory, 期間)

    current_cost = calculate_total_cost(current_sol, initial_inventory)
    best_sol = [row[:] for row in current_sol]
    best_cost = current_cost
    temp = initial_temp
    
    start_time = time.time()
    
    for it in range(max_iterations):
        new_sol = generate_neighbor(current_sol, initial_inventory, 期間)
        new_cost = calculate_total_cost(new_sol, initial_inventory)
        
        if new_cost < current_cost:
            current_sol = new_sol
            current_cost = new_cost
            if current_cost < best_cost:
                best_sol = [row[:] for row in current_sol]
                best_cost = current_cost
        else:
            delta = new_cost - current_cost
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_sol = new_sol
                current_cost = new_cost
        
        temp *= cooling_rate
        if temp < 0.1: break
        
        if it % 50000 == 0:
            print(f"  Iter {it}: Best Cost {int(best_cost)} (Temp {int(temp)})")
            
    return best_sol, best_cost

# ==========================================
# ラッパー・実行関数
# ==========================================

def solve_mip(initial_inventory_list_arg, time_limit=500):
    import pulp
    model = pulp.LpProblem("ProductionScheduling", pulp.LpMinimize)
    品番数 = len(品番リスト)
    期間 = 20
    品目 = range(品番数); 期間_index = range(期間)
    Production = pulp.LpVariable.dicts("Production", (品目, 期間_index), lowBound=0, cat='Integer')
    IsProduced = pulp.LpVariable.dicts("IsProduced", (品目, 期間_index), cat='Binary')
    Inventory = pulp.LpVariable.dicts("Inventory", (品目, 期間_index), lowBound=0, cat='Continuous')
    Shortage = pulp.LpVariable.dicts("Shortage", (品目, 期間_index), lowBound=0, cat='Continuous')
    WorkTime = pulp.LpVariable.dicts("WorkTime", 期間_index, lowBound=0, cat='Continuous')
    Overtime = pulp.LpVariable.dicts("Overtime", 期間_index, lowBound=0, cat='Continuous')

    total_cost = pulp.lpSum(在庫コスト単価 * Inventory[i][t]/収容数リスト[i] for i in 品目 for t in 期間_index) + \
                 pulp.lpSum(残業コスト単価 * Overtime[t] for t in 期間_index) + \
                 pulp.lpSum(段替えコスト単価 * IsProduced[i][t] for i in 品目 for t in 期間_index) + \
                 pulp.lpSum(出荷遅れコスト単価 * Shortage[i][t]/収容数リスト[i] for i in 品目 for t in 期間_index)
    model += total_cost
    bigM = 1000000
    for i in 品目:
        for t in 期間_index:
            d = get_demand(i, t)
            prev_inv = initial_inventory_list_arg[i] if t == 0 else (Inventory[i][t-1] - Shortage[i][t-1])
            model += Inventory[i][t] - Shortage[i][t] == prev_inv + Production[i][t] - d
            model += Production[i][t] <= bigM * IsProduced[i][t]
    for t in 期間_index:
        model += WorkTime[t] == pulp.lpSum(Production[i][t] * (サイクルタイムリスト[i] / 込め数リスト[i]) + 段替え時間 * IsProduced[i][t] for i in 品目)
        model += WorkTime[t] <= 定時 + Overtime[t]
        model += WorkTime[t] <= 定時 + 最大残業時間
        model += Overtime[t] >= WorkTime[t] - 定時
        model += Overtime[t] >= 0
        model += pulp.lpSum(IsProduced[i][t] for i in 品目) <= 5
    try:
        solver = pulp.GUROBI(msg=True, timelimit=time_limit)
        # If Gurobi isn't actually available, raise to trigger fallback
        if hasattr(solver, 'available') and not solver.available():
            raise RuntimeError('Gurobi not available')
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
    try:
        model.solve(solver)
    except pulp.PulpSolverError:
        # If the chosen solver fails at solve time (e.g. Gurobi license issue),
        # fallback to CBC once before giving up.
        if not isinstance(solver, pulp.PULP_CBC_CMD):
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
            model.solve(solver)
        else:
            raise
    status = pulp.LpStatus[model.status]
    if status in ['Optimal', 'Not Solved'] and pulp.value(model.objective) is not None:
        sch = [[0]*期間 for _ in range(品番数)]
        for i in 品目:
            for t in 期間_index: sch[i][t] = pulp.value(Production[i][t])
        return sch, pulp.value(model.objective), {}
    return None, None, {}

def multi_start_simulated_annealing(initial_inventory, num_starts=3, **sa_params):
    best_sol = None; best_val = float('inf')
    print(f"=== SA Optimization (Runs: {num_starts}) ===")
    
    for s in range(num_starts):
        print(f"Run {s+1}/{num_starts} started...")
        sol, val = simulated_annealing_scheduler(initial_inventory, **sa_params)
        print(f"Run {s+1} finished. Cost: {int(val)}")
        
        if val < best_val:
            best_sol = sol; best_val = val
            print(f"-> New Best Found!")
            
    if best_sol:
        shorts = calculate_shortage_by_item(best_sol, initial_inventory)
        s_items = sorted([(i, v) for i, v in enumerate(shorts) if v > 0], key=lambda x: x[1], reverse=True)
        if s_items:
            print("\n!!! WARNING: Remaining Shortages !!!")
            for idx, qty in s_items[:5]:
                print(f"  {品番リスト[idx]}: {qty:.1f}")
        else:
            print("\n>>> No Shortages Detected. <<<")
            
    return best_sol, best_val

def calculate_shortage_by_item(production_schedule, initial_inventory, 期間=20):
    shorts = [0.0]*len(initial_inventory)
    inv = list(initial_inventory)
    for t in range(期間):
        for i in range(len(initial_inventory)):
            inv[i] += production_schedule[i][t]
            d = get_demand(i, t)
            if inv[i] < d:
                shorts[i] += d - inv[i]
                inv[i] = 0
            else: inv[i] -= d
    return shorts

def save_results_to_csv(results, filename):
    os.makedirs('result', exist_ok=True)
    df = pd.DataFrame(results)
    path = f'result/{filename}'
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return path

def calculate_summary_statistics(results_data, title="統計"):
    if not isinstance(results_data, pd.DataFrame): results_data = pd.DataFrame(results_data)
    if results_data.empty: return None, None
    nums = results_data.select_dtypes(include=[np.number])
    stats = {'項目': nums.columns, '平均': nums.mean().values, '最小': nums.min().values, '最大': nums.max().values}
    df = pd.DataFrame(stats)
    print(f"\n=== {title} ===\n{df.to_string(index=False)}")
    save_results_to_csv(df.to_dict('records'), f'{title}_stats.csv')
    return df, None

def save_schedule_details(production_schedule, initial_inventory, file_name, method_name="", 期間=20):
    os.makedirs('result', exist_ok=True)
    base_path = f'result/{method_name}_{file_name}'
    品番数 = len(品番リスト)
    cols = [f'{t+1}日目' for t in range(期間)]
    
    df_prod = pd.DataFrame(production_schedule, index=品番リスト, columns=cols)
    df_prod.insert(0, '品番', 品番リスト)
    df_prod.to_csv(f'{base_path}_production.csv', index=False, encoding='utf-8-sig')
    
    inventory_matrix = []
    shortage_matrix = []
    daily_stats = []
    current_inventory = list(initial_inventory)
    
    for t in range(期間):
        daily_work = 0; daily_setup = 0; daily_prod_time = 0
        col_inv = []; col_short = []
        for i in range(品番数):
            amount = production_schedule[i][t]
            if amount > 0:
                s = 段替え時間
                p = (amount / 込め数リスト[i]) * サイクルタイムリスト[i]
                daily_setup += s; daily_prod_time += p; daily_work += s + p
            current_inventory[i] += amount
            d = get_demand(i, t)
            if current_inventory[i] >= d:
                current_inventory[i] -= d
                col_short.append(0)
            else:
                col_short.append(d - current_inventory[i])
                current_inventory[i] = 0
            col_inv.append(current_inventory[i])
        inventory_matrix.append(col_inv)
        shortage_matrix.append(col_short)
        
        overtime = max(0, daily_work - 定時)
        daily_stats.append({
            '日': t + 1, '総稼働時間': daily_work, '定時内': min(daily_work, 定時),
            '残業時間': overtime, '段替え時間': daily_setup, '生産時間': daily_prod_time,
            '違反時間': max(0, daily_work - (定時 + 最大残業時間))
        })

    df_inv = pd.DataFrame(np.array(inventory_matrix).T, index=品番リスト, columns=cols)
    df_inv.insert(0, '品番', 品番リスト)
    df_inv.to_csv(f'{base_path}_inventory.csv', index=False, encoding='utf-8-sig')
    
    df_short = pd.DataFrame(np.array(shortage_matrix).T, index=品番リスト, columns=cols)
    df_short.insert(0, '品番', 品番リスト)
    df_short.to_csv(f'{base_path}_shortage.csv', index=False, encoding='utf-8-sig')
    
    df_stats = pd.DataFrame(daily_stats)
    df_stats.to_csv(f'{base_path}_daily_stats.csv', index=False, encoding='utf-8-sig')
    print(f"詳細データをCSV保存しました: {base_path}_*.csv")

def plot_result(production_schedule, initial_inventory, file_name, method_name="", 期間=20):
    save_schedule_details(production_schedule, initial_inventory, file_name, method_name, 期間)

    品番数 = len(品番リスト)
    working_times = []; setup_times = []; production_times = []
    inventory_by_item = []; production_by_item = []; shortage_by_item = []
    current_inventory = initial_inventory[:]
    max_work = 定時 + 最大残業時間

    for t in range(期間):
        daily_work = 0; daily_setup = 0; daily_prod = 0
        p_inv = []; p_short = []; p_prod = []
        for i in range(品番数):
            amount = production_schedule[i][t] if production_schedule else 0
            p_prod.append(amount)
            if amount > 0:
                s = 段替え時間
                p = (amount / 込め数リスト[i]) * サイクルタイムリスト[i]
                daily_setup += s; daily_prod += p; daily_work += s + p
            current_inventory[i] += amount
            d = get_demand(i, t)
            if current_inventory[i] >= d:
                current_inventory[i] -= d
                p_short.append(0)
            else:
                p_short.append(d - current_inventory[i])
                current_inventory[i] = 0
            p_inv.append(current_inventory[i])
        working_times.append(daily_work)
        setup_times.append(daily_setup)
        production_times.append(daily_prod)
        inventory_by_item.append(p_inv)
        shortage_by_item.append(p_short)
        production_by_item.append(p_prod)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    periods = range(1, 期間 + 1)
    colors = plt.cm.Set3(np.linspace(0, 1, 品番数))

    ax1.bar(periods, setup_times, label='Setup', color='coral', alpha=0.8)
    ax1.bar(periods, production_times, bottom=setup_times, label='Prod', color='skyblue', alpha=0.8)
    ax1.axhline(y=定時, color='g', ls='--'); ax1.axhline(y=max_work, color='r', ls='--')
    ax1.set_title(f'稼働時間', fontsize=18, fontweight='bold'); ax1.legend(fontsize=16, loc='upper right')
    ax1.set_xticks(range(1, 期間+1))
    ax1.set_xticklabels([str(i) for i in range(1, 期間+1)], fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    if PLOT_YLIM_WORKING: ax1.set_ylim(PLOT_YLIM_WORKING)
    else: ax1.set_ylim(0, max_work + 60)

    inv_data = np.array(inventory_by_item).T
    bot = np.zeros(期間)
    for i in range(min(10, 品番数)):
        ax2.bar(periods, inv_data[i], bottom=bot, label=str(i), color=colors[i], alpha=0.8)
        bot += inv_data[i]
    if 品番数 > 10: ax2.bar(periods, np.sum(inv_data[10:], axis=0), bottom=bot, label='Others', color='gray')
    ax2.set_title('在庫量', fontsize=18, fontweight='bold'); ax2.legend(fontsize=16, loc='upper right')
    ax2.set_xticks(range(1, 期間+1))
    ax2.set_xticklabels([str(i) for i in range(1, 期間+1)], fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    if PLOT_YLIM_INVENTORY: ax2.set_ylim(PLOT_YLIM_INVENTORY)

    prod_data = np.array(production_by_item).T
    bot = np.zeros(期間)
    for i in range(min(10, 品番数)):
        ax3.bar(periods, prod_data[i], bottom=bot, label=str(i), color=colors[i], alpha=0.8)
        bot += prod_data[i]
    ax3.set_title('生産量', fontsize=18, fontweight='bold'); ax3.legend(fontsize=16)
    ax3.set_xticks(range(1, 期間+1))
    ax3.set_xticklabels([str(i) for i in range(1, 期間+1)], fontsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    if PLOT_YLIM_PRODUCTION: ax3.set_ylim(PLOT_YLIM_PRODUCTION)

    short_data = np.array(shortage_by_item).T
    bot = np.zeros(期間)
    for i in range(min(10, 品番数)):
        ax4.bar(periods, short_data[i], bottom=bot, label=str(i), color=colors[i], alpha=0.8)
        bot += short_data[i]
    ax4.set_title('出荷遅れ量', fontsize=18, fontweight='bold'); ax4.legend(fontsize=16, loc='upper right')
    ax4.set_xticks(range(1, 期間+1))
    ax4.set_xticklabels([str(i) for i in range(1, 期間+1)], fontsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    if PLOT_YLIM_SHORTAGE: ax4.set_ylim(PLOT_YLIM_SHORTAGE)

    plt.tight_layout()
    os.makedirs('result', exist_ok=True)
    plt.savefig(f'result/{method_name}_results_{file_name}.png', dpi=300)
    print(f"Plot saved: result/{method_name}_results_{file_name}.png")
    
    return sum(1 for w in working_times if w > max_work)