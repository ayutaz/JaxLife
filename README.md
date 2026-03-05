# JaxLife - Jax によるオープンエンドな人工生命シミュレータ

## 🧬 JaxLife とは
JaxLife は、複雑で高度な行動の進化に焦点を当てた人工生命シミュレータです。
エージェントが Turing-complete な計算を実行できるロボットシステムと相互作用することで、これを実現しています。


### 💻 シミュレーション
本シミュレーションは、*エージェント*、*ロボット*、*地形* で構成されています。エージェントは自然選択によって進化し、リカレントニューラルネットワークによってパラメータ化されています。ロボットはエージェントによってプログラムでき、Turing-complete な計算を実行することが可能です。
地形はシミュレーションの多くの側面（移動のしやすさ、利用可能なエネルギー量など）を制御します。また、天候や気候に似たシステムによって、地形はゆっくりと変化します。
<p align="center">
    <img src="pics/fig1.png">
</p>

### 👾 エージェント
エージェントのアーキテクチャは、近くのすべてのエンティティに対して個別のエンコーダを使用します。これらは Self Attention で処理された後、エージェント自身の埋め込みとの Cross Attention が行われます。地形は別のエンコーダで処理され、最終的なエンティティ表現と結合されます。これらすべてが LSTM に通されます。

エージェントは自然選択によって進化し、小さなランダムな摂動を突然変異として加えながら、自身の重みを子孫に引き継ぎます。
エージェントは、移動、食事、地形変更、攻撃、ロボットのプログラミングなど、さまざまな行動を実行できます。
<p align="center">
    <img src="pics/agents.png">
</p>

### 🤖 ロボット
ロボットはエージェントと同じ行動空間を持ち、メッセージを送信することでプログラムできます。
これらのロボットは、理論的に Turing-complete な計算を実行できます。
<p align="center">
    <img src="pics/rule110.gif">
</p>

また、輸送、農業、通信などの実用的なタスクも実行できます。

<p align="center">
    <img width="30%" src="pics/train.gif">
    <img width="30%" src="pics/oscillate.gif">
    <img width="30%" src="pics/terraform.gif">
</p>

## 📝 結果
シミュレーションを実行すると、エージェントとロボットが大規模な地形変更を行ったり、行動を協調させたりするなど、興味深い創発的特性を観察できます。
<p align="center">
    <img src="pics/bridge.gif">
</p>

## ✍️ 使い方

### 🛠️ インストール

```bash
git clone https://github.com/luchris429/JaxLife.git
cd JaxLife
uv sync
uv run pre-commit install
```

### 🏃 実行方法
以下のコマンドでシミュレーションを開始できます。さまざまな設定オプションがあります。詳細は `src/main.py` を参照してください。
主要なパラメータは以下の通りです：
- `--gui`: Pygame を使用してインタラクティブに実行
- `--wandb`: Weights and Biases へのログ記録の有無
- `--num-agents`: シミュレーション内のエージェント数
```bash
uv run python src/main.py
```


## 🔍 関連プロジェクト

- [Alien](https://github.com/chrxh/alien): CUDA を活用した人工生命シミュレーション
- [Leniax](https://github.com/morgangiraud/leniax): Jax による [Lenia](https://chakazul.github.io/lenia.html#Code) の実装