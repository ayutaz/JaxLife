# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

JaxLifeはJAXベースの人工生命シミュレータ。エージェントは自然選択により進化し、リカレントニューラルネットワーク（LSTM）でパラメータ化されている。エージェントはチューリング完全な計算が可能なプログラム可能ロボットと相互作用する。シミュレーションは天候・気候システムを持つ地形グリッド上で実行される。

## シミュレーションの実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# シミュレーション実行（リポジトリルートから）
python src/main.py

# インタラクティブGUIモード
python src/main.py --gui

# 主要フラグ
#   --wandb              Weights & Biasesへのログ記録
#   --num-agents N       エージェント数（デフォルト: 128）
#   --num-bots N         ロボット数（デフォルト: 32）
#   --brain TYPE         エージェント脳タイプ: lstm_brain, zero_brain, random_brain, debug_brain, profiling_brain
#   --terrain-init TYPE  地形: perlin, random, four_islands, two_islands, *_weather系
#   --map-size N         グリッドサイズ（デフォルト: 128）
#   --no-render          レンダリング無効化
```

このプロジェクトにテストはない。

## アーキテクチャ

### コアループ

`src/main.py`で全設定をフラットなdictとして定義し、メインループを実行する。各外側ステップで`jax.lax.scan`を`NUM_WORLD_STEPS`（512）回の内側ステップに対して実行する。`World.step()`の処理順序: **エージェント更新 -> ボット更新（x4） -> 地形更新**。

### 主要モジュール

- **`src/world/world.py` (World)**: シミュレーション全体の統括。各ステップで全エージェント/ボットに対して`vmap`で状態差分を計算し、差分を合計してワールド状態に適用する。この差分ベースのアプローチがJAXの並列化を可能にする。
- **`src/world/structs.py`**: `flax.struct.dataclass`による全状態型の定義: `WorldState`, `AgentState`, `BotState`, `TerrainState`, `Action`, 観測。`WorldState.clone_no_params()`はvmap操作時のメモリ削減のためニューラルネットワークパラメータを除去する。
- **`src/world/agent.py` (Agent)**: エージェントロジック — 観測収集（最近傍エージェント/ボット + 地形グリッド）、行動実行（移動、捕食、攻撃、メッセージング、テラフォーミング、繁殖）。繁殖時にゲノムパラメータの突然変異を適用。
- **`src/world/bots.py` (Bot)**: ロボットロジック — エージェントと同様の行動空間だが、ニューラルネットワークではなく決定的プログラムで駆動される。ロボットは地形ビットの読み書きが可能。
- **`src/world/terrain.py` (Terrain)**: 地形初期化戦略のファクトリ。

### 脳システム

- **エージェント脳** (`src/agent_brains/`): `LSTMBrain`が主要な脳 — エンティティ埋め込みに対するself-attention、エージェント自身の埋め込みとのcross-attention、地形エンコーディングを全てLSTMに入力する。他の脳（zero, random, debug, profiling）はテスト用。
- **ロボット脳** (`src/robot_brains/`): `RobotComplexProgramBrain`は7つの演算とルックアップテーブルを持つプログラム可能な計算システムを実装し、チューリング完全な動作を可能にする。

### 設計パターン

- **差分ベースの状態更新**: 行動はゼロ初期化された状態差分を生成し、全エンティティで合計した後、現在の状態に加算する。これにより変更を避け、JAXの関数型パラダイムに適合する。
- **`tree_where`** (`src/utils.py`): マスクに基づいて2つのpytreeを条件選択するユーティリティ。形状の自動ブロードキャスト付き。
- **dictによる設定管理**: 全設定はフラットなPython dict（dataclass/YAMLではない）で渡される。キーはUPPER_SNAKE_CASE。
- **トロイダルワールド**: 位置は剰余演算（`MAP_SIZE * CELL_SIZE`）で折り返す。
- **観測**: エージェントは最近傍N体のエンティティ（距離順）とローカル地形グリッドを観測する。
