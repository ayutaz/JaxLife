# JaxLife シミュレーションアルゴリズム詳細

## 目次

1. [全体概要](#全体概要)
2. [シミュレーションループ](#シミュレーションループ)
3. [状態更新メカニズム（差分ベースアーキテクチャ）](#状態更新メカニズム差分ベースアーキテクチャ)
4. [エージェントシステム](#エージェントシステム)
   - [ニューラルネットワーク（LSTM Brain）](#ニューラルネットワークlstm-brain)
   - [観測の収集](#観測の収集)
   - [行動空間](#行動空間)
   - [エネルギーコスト計算](#エネルギーコスト計算)
5. [自然選択と進化](#自然選択と進化)
   - [繁殖条件](#繁殖条件)
   - [突然変異](#突然変異)
   - [集団管理](#集団管理)
6. [ロボット（Bot）システム](#ロボットbotシステム)
   - [プログラム可能な脳](#プログラム可能な脳)
   - [地形ビット読み書き](#地形ビット読み書き)
7. [地形・天候システム](#地形天候システム)
   - [Perlin ノイズによる地形生成](#perlin-ノイズによる地形生成)
   - [天候変動アルゴリズム](#天候変動アルゴリズム)
   - [地形のベース状態への回帰](#地形のベース状態への回帰)
8. [ワールドの物理法則](#ワールドの物理法則)
9. [計算最適化（JAX による並列化）](#計算最適化jax-による並列化)

---

## 全体概要

JaxLife は **自然選択** によって駆動される人工生命シミュレータである。中核となるアイデアは以下の通り：

- **エージェント**: リカレントニューラルネットワーク（LSTM）を「ゲノム」として持つ自律的存在。エネルギーが尽きると死亡し、十分なエネルギーを蓄えると繁殖できる。子はゲノム（ニューラルネットワークの重み）を突然変異付きで受け継ぐ。
- **ロボット**: エージェントによってプログラム可能な決定的オートマトン。チューリング完全な計算が可能。
- **地形**: 各セルがエネルギー量や移動コストなどの属性を持つ2Dグリッド。天候システムにより緩やかに変動する。

シミュレーション全体は純粋関数型で実装されており、JAX の `jit`、`vmap`、`lax.scan` を活用して GPU 上で高速に並列実行される。

---

## シミュレーションループ

```
外側ループ (NUM_OUTER_STEPS = 128回)
  └── 内側ループ (jax.lax.scan, NUM_WORLD_STEPS = 512回)
        └── World.step() 1回の処理:
              1. agent_update()    — 全エージェントの行動実行
              2. bot_update() x4   — ロボットはエージェントの4倍の頻度で動作
              3. terrain_update()  — 地形と天候の更新
              4. t += 1
```

`jax.lax.scan` はループ展開なしに固定回数の反復を JIT コンパイルする JAX のプリミティブで、XLA による最適化が可能になる。外側ループでは各スキャンの完了後にメトリクスのログとレンダリングを行う。

GUI モードでは `lax.scan` を使わず、1ステップずつ `world.step` を呼び出してリアルタイム描画する。

---

## 状態更新メカニズム（差分ベースアーキテクチャ）

JaxLife の最も重要な設計パターンは **差分ベースの状態更新** である。JAX は副作用のない純粋関数を前提とするため、配列のインプレース更新はできない。この制約の下で複数のエージェントが同時にワールド状態を変更するために、以下の手法を使用する：

### アルゴリズム

```
1. world_state_diff_zero = zeros_like(world_state)  # ゼロ初期化された差分
2. 各エージェント i について（vmap で並列実行）:
   a. 観測を収集
   b. 脳（NN）で行動を決定
   c. 行動から world_state_diff[i] を計算
      - 自分のエネルギーの変化、位置の変化
      - 他のエージェント/ボットへの影響（押す、攻撃など）
      - 地形への影響（テラフォーミング、食事）
   d. alive でない場合は diff をゼロに (tree_where)
3. world_state_new = sum(world_state_diff, axis=0) + world_state
4. clip_state() で値を有効範囲にクリップ
```

`tree_where` は pytree 全体に対して条件マスクを適用するユーティリティで、死亡エージェントの差分をゼロにするために使われる。

この差分の合計アプローチにより、複数エージェントが同一の地形セルやエージェントに同時に影響を与える場合でも、全ての影響が加算的に反映される。

---

## エージェントシステム

### ニューラルネットワーク（LSTM Brain）

エージェントの「脳」は以下の構造を持つ Flax ベースのニューラルネットワークである：

```
入力:
  ├── 近傍エージェント (AGENT_NUM_VIEW_AGENTS = 16体)
  │     └── Dense(HSIZE) → tanh → Self-Attention (4 heads)
  ├── 近傍ロボット (AGENT_NUM_VIEW_BOTS = 2体)
  │     └── Dense(HSIZE) → tanh → Self-Attention (4 heads) [エージェントと結合]
  ├── 自身の状態
  │     └── Dense(HSIZE) → tanh
  └── 地形グリッド (9x9 = 81セル)
        └── Dense(HSIZE) → flatten → Dense(HSIZE)

処理:
  1. エンティティ埋め込み = [エージェント埋め込み, ボット埋め込み]
  2. Self-Attention で全エンティティ間の関係を計算
  3. Cross-Attention: 自身の埋め込み(Q) × エンティティ(K,V)
  4. 地形エンコーディングと結合
  5. LSTM セルで時系列処理 (carry が hidden state)

出力:
  Dense(HSIZE) → tanh → Dense(action_size) → tanh → logits_to_actions
```

- **隠れ状態サイズ**: HSIZE = 64
- **出力サイズ**: 6 + MESSAGE_SIZE * 2 + 5 = 43 次元
- **注意**: 学習（勾配降下）は行わない。重みは突然変異によってのみ変化する。

### 観測の収集

各エージェントは以下を観測する：

1. **近傍エージェント**: ユークリッド距離で最も近い N 体（デフォルト 16 体）。各エージェントの観測値は `[log(energy), log(age), dx, dy, last_action, id, self_message, other_message]` のベクトル。
2. **近傍ロボット**: 最も近い 2 体。`[log(energy), dx, dy, memory, program]`。
3. **地形グリッド**: 自身を中心とした 9x9 のグリッド（TERRAIN_VIEW_RANGE = 4）。各セルの属性 7 次元。
4. **自身の状態**: 自分自身のエージェント観測ベクトル。

距離計算はトロイダルワールド（端が反対側につながる）を考慮し、`min(diff, MAP_SIZE * CELL_SIZE - diff)` で計算される。死亡エージェントは距離にペナルティを加算して非表示にする。

### 行動空間

エージェントが出力する行動は 15 種類のフィールドで構成される：

| 行動 | 型 | 説明 |
|---|---|---|
| `x_move`, `y_move` | float [-1, 1] | 移動方向。`MOVE_SPEED` (0.5) を乗算 |
| `push` | float [-1, 1] | 近傍エンティティを押す/引く |
| `eat` | bool (>0で実行) | 地形のエネルギーを食べる |
| `reproduce` | bool (>0で実行) | 繁殖（条件を満たす場合のみ） |
| `hit` | float [-1, 1] | 攻撃。距離減衰あり |
| `message_other` | float (ReLU) | メッセージ送信の強度 |
| `self_message` | float[16] | 自身の記憶への差分加算 |
| `other_message` | float[16] | 他者へ送信するメッセージ内容 |
| `terrain_*` (5種) | float | 地形パラメータの変更（テラフォーミング） |
| `read_terrain_bit` | float | ロボット専用：地形ビットの読み書き |

### エネルギーコスト計算

```python
energy_cost = (
    MOVE_COST * terrain.move_cost * |x_move|
  + MOVE_COST * terrain.move_cost * |y_move|
  + PUSH_COST * terrain.push_cost * |push|
  + MESSAGE_COST * terrain.message_cost * message_other
  + MAX_REPRODUCE_COST * agent_frac * terrain.reproduce_cost * reproduce
  + HIT_COST * hit
  + TERRAIN_MOD_COST * (terrain_energy_gain + terrain_move_cost + ...)
  + life_cost  # = age / AGE_COST + LIFE_COST
)
eat_gain = EAT_RATE * terrain.energy_amt * eat

net_energy_change = -energy_cost + eat_gain
```

重要な点：
- **移動コストは地形依存**: `terrain.move_cost` により、地形ごとに移動の難易度が異なる
- **生存コストは年齢に比例**: `age / AGE_COST + LIFE_COST` で、老化するほどコストが増大
- **繁殖コストは密度依存**: `agent_frac`（生存エージェント比率）が高いほどコスト増大
- **エネルギーが 0 以下になると死亡**

---

## 自然選択と進化

### 繁殖条件

```python
can_reproduce = (
    action.reproduce > 0                                    # 繁殖行動を選択
    AND energy > MAX_REPRODUCE_COST * terrain.reproduce_cost  # 十分なエネルギー
                * REPRODUCE_FACTOR * agent_frac               # 密度に依存
    AND alive                                                # 生存中
)
```

`REPRODUCE_FACTOR = 2` により、繁殖コストの 2 倍のエネルギーを保持していなければ繁殖できない。これは親が繁殖直後に死亡することを防ぐ。

### 突然変異

子エージェントのゲノム（ニューラルネットワークの全重み）に対して、ガウスノイズを加算する：

```python
child_genome = parent_genome + N(0, MUTATION_STD)  # MUTATION_STD = 0.01
child_id = clip(parent_id + U(-1, 1), -1, 1)       # IDも摂動
child_hidden_state = fresh_initialization            # LSTMの隠れ状態はリセット
child_energy = INITIAL_ENERGY                        # エネルギーは初期値
child_position = parent_position                     # 位置は親と同じ
```

突然変異は全パラメータに一様に適用される。この単純な進化戦略（ES）で、有利な行動パターンを持つニューラルネットワークの重みが集団内で広まる。

### 集団管理

1. **エネルギー降順ソート**: 繁殖判定前にエージェントをエネルギー降順にソート
2. **繁殖スロットの割り当て**: 繁殖条件を満たすエージェントの子を生成（`parent_idxs`）
3. **エネルギー昇順ソート（反転）**: 死亡エージェントが先頭に来るようにする
4. **空きスロットへの配置**: 死亡エージェントのスロットに新しい子を `tree_where` で上書き
5. **最低人口保証**: 全エージェントが死滅した場合（`ONLY_RESET_WHEN_NO_AGENTS = True`）、`MIN_AGENTS` 体のランダムエージェントを生成

---

## ロボット（Bot）システム

ロボットはエージェントの 4 倍の頻度で行動する（`BOT_STEP_RATIO = 4`）。

### プログラム可能な脳

`RobotComplexProgramBrain` は 7 つの演算と 8 エントリのルックアップテーブルで構成される：

```
入力: program[16], self_message (memory)[16], other_messages[2][16]

演算の選択: argmax(program[:7])  →  7つの演算のうち1つを実行
ルックアップテーブル: program[7:15]

演算:
  0: COPY     → m1 をメモリにコピー（常に self_message を書き込み）
  1: IDENTITY → mem をそのまま使用
  2: MUL      → mem * m1
  3: MMA      → mem * m1 + m2
  4: XOR_MUL  → XOR(mem[-1], m1[-1]) に基づく条件付き乗算
  5: NAND     → 1 - (m1 * m2)
  6: LOOKUP   → 3ビット入力 (m1[-1], mem[-1], m2[-1]) でルックアップテーブル参照
```

演算 6（ルックアップテーブル）は任意の 3 入力 1 出力ブール関数を表現できるため、NAND を含む全論理ゲートを構成可能。これとメモリ（`self_message`）の組み合わせにより、理論上チューリング完全な計算が可能になる。

エージェントは `message_other` 行動でロボットの `program` フィールドを書き換えることができるため、エージェントがロボットを「プログラム」できる。

### 地形ビット読み書き

ロボットは地形の `bits` フィールド（各セルに 1 ビット）を読み書きできる：

```python
# 書き込み (read_terrain_bit < -0.99):
terrain.bits[x, y] = memory[-1]

# 読み取り (read_terrain_bit > +0.99):
memory[-1] = terrain.bits[new_x, new_y]  # 移動先の地形ビットを読む
```

この機構により、ロボットは地形を「テープ」として使う形でチューリングマシンのような計算を実行できる（例：Rule 110 セルオートマトンの実装）。

---

## 地形・天候システム

### Perlin ノイズによる地形生成

地形は **Perlin ノイズ** で生成される。Perlin ノイズはコンピュータグラフィクスで広く使われる手続き的ノイズ生成アルゴリズムで、自然な見た目の地形を生成する。

#### Perlin ノイズのアルゴリズム

```
1. 低解像度グリッド (PERLIN_RES × PERLIN_RES = 4×4) の各頂点にランダムな角度を割り当て
2. 角度から勾配ベクトル (cos(θ), sin(θ)) を計算
3. 各ピクセルについて、周囲4つの勾配ベクトルとの内積を計算
4. 5次補間関数 t³(6t² - 15t + 10) で滑らかに補間
5. [0, 1] に正規化
```

地形の 7 つの属性（energy_amt, move_cost, push_cost, energy_gain, message_cost, reproduce_cost, max_energy）はそれぞれ独立した Perlin ノイズマップから生成される。生成は `jax.vmap` で 7 マップ同時に並列化される。

各属性は設定の最大値でスケーリングされる：
- `move_cost`: 0 〜 MAX_TERRAIN_MOVE_COST (4.0)
- `energy_gain`: 0 〜 MAX_TERRAIN_ENERGY_GAIN (1.0)
- `max_energy`: 0 〜 MAX_TERRAIN_ENERGY (8.0)

### 天候変動アルゴリズム

地形は時間経過とともに緩やかに変動する。これは「天候」として実装されている：

```
毎ステップ:
  1. noise_angles += U(0, WEATHER_CHANGE_SPEED)  # 角度を微小に摂動
     WEATHER_CHANGE_SPEED = 0.001

  2. base_terrain = regenerate_from_angles(noise_angles)
     # 新しい角度から Perlin ノイズを再生成
```

Perlin ノイズの入力角度を少しずつ変化させることで、地形が滑らかに移り変わる。角度の変化が小さいため、地形の変動は非常にゆっくり進行する。

### 地形のベース状態への回帰

実際の地形（エージェントの行動による変更を含む）は、ベース状態に向かってゆっくり回帰する：

```python
regress_speed = base_terrain.max_energy / MAX_TERRAIN_ENERGY * TERRAIN_ALPHA
# TERRAIN_ALPHA = 0.005

new_terrain = regress_speed * base_terrain + (1 - regress_speed) * current_terrain
```

これにより：
- エージェントがテラフォーミングした地形は、時間経過で元に戻る
- 回帰速度は `max_energy` に比例（豊かな地形ほど回復が速い）
- `TERRAIN_ALPHA = 0.005` により、回帰は非常にゆっくり

エネルギーの成長は別の式で計算される：

```python
energy_amt += energy_gain - MAX_TERRAIN_ENERGY_GAIN / TERRAIN_GAIN_SCALING
```

`TERRAIN_GAIN_SCALING = 1.5` により、`energy_gain` が `MAX_TERRAIN_ENERGY_GAIN / 1.5 ≈ 0.667` を超えるセルのみでエネルギーが正味で増加する。

---

## ワールドの物理法則

### トロイダルワールド

ワールドは端が接続されたトーラス構造：

```python
pos_x = pos_x % (MAP_SIZE * CELL_SIZE)  # = 128 * 8 = 1024
pos_y = pos_y % (MAP_SIZE * CELL_SIZE)
```

### 攻撃の距離減衰

```python
hit_damage = hit * HIT_STRENGTH * exp(-distance * HIT_DISTANCE_DECAY)
```

- `HIT_STRENGTH = 1.0`
- `HIT_DISTANCE_DECAY = 0.5`
- 近いほどダメージが大きい指数減衰モデル

### 状態クリッピング

各ステップの終了時に以下の制約が適用される：

| フィールド | 制約 |
|---|---|
| `energy_amt` | [0, max_energy] |
| `move_cost`, `push_cost` 等 | [0, ∞) |
| `energy_gain` | [0, 2.0] |
| `agent.energy` | [0, ∞)（0以下で死亡） |
| `agent.pos` | トロイダル折り返し |
| `agent.self_message` | tanh |
| `bot.program`, `bot.memory` | [-1, 1] |

### Kardashev スコア

全エージェントとロボットのエネルギー消費の合計が「Kardashev スコア」として追跡される。これはシミュレーション内の文明の活動レベルの指標として機能する（カルダシェフ・スケールに着想）。

---

## 計算最適化（JAX による並列化）

### vmap による並列化

全エージェント/ボットの行動は `jax.vmap` で並列に計算される：

```python
# 全エージェントを同時に実行
world_state_diff, ... = jax.vmap(agent.act, in_axes=(0, 0, None))(
    agent_states,           # バッチ軸 0
    jnp.arange(NUM_AGENTS), # インデックス
    world_state_no_params,  # ブロードキャスト（全エージェント共通）
)
```

### メモリ最適化

`WorldState.clone_no_params()` は `genome_params` と `hidden_state_params` を `None` に置換する。vmap 内ではこれらの巨大なパラメータは個別のエージェント状態から参照されるが、世界状態の共有コピーからは不要なためである。

### 差分の合計

```python
world_state_new = jax.tree.map(
    lambda diff, state: diff.sum(axis=0) + state,
    world_state_diff,    # shape: (NUM_AGENTS, ...)
    world_state          # shape: (...)
)
```

`tree.map` により、pytree の全リーフに対して同一の操作を一括適用する。

### Saliency 計算

エージェントの観測に対するネットワーク出力の勾配を `jax.grad` で計算し、どの入力がエージェントの行動に最も影響するかを分析する。これは通信（メッセージ）の発達を追跡するために使用される。

---

## まとめ

JaxLife のアルゴリズムは、以下の要素の組み合わせによって複雑な創発的行動を生み出す：

1. **進化的ニューラルネットワーク**: 勾配降下なしに、自然選択と突然変異のみでニューラルネットワークの重みを最適化
2. **差分ベースの並列状態更新**: JAX の関数型パラダイムに適合しつつ、全エージェントの行動を並列に計算・統合
3. **Perlin ノイズ天候システム**: 地形の緩やかな変動により、適応圧を維持
4. **チューリング完全なロボットシステム**: エージェントが道具（ロボット）をプログラムする能力を進化させる余地を提供
5. **密度依存のフィードバック**: 繁殖コストの密度依存性により、集団サイズが自己調整される

これらの組み合わせにより、協調行動、大規模テラフォーミング、ロボットの道具利用など、手動でプログラムされたものではない創発的な複雑行動が観察される。
