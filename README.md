# reachpath 海鲜经营资料服务

这个 Python 项目为经营研究提供本地 CSV 资料、领域视图、数值剖面和不可变研究备注。现有经营资料的字段与计量语义见 `docs/domain.md`，HTTP 行为见 `docs/api.md`。服务不连接真实经营系统，也不自动执行交易或订货。

## 目录

`reachpath/catalog.py` 解析并校验资料，`domain.py` 提供当前经营视图，`profile.py` 提供探索性数值摘要，`paths.py` 实施随机分组下的路径（中介）分析，`cases.py` 保存含资料副本的研究备注，`service.py` 处理 HTTP。`data/` 是可重复使用的虚构经营资料，`tests/` 检查当前接口和领域口径，`smoke.py` 发起真实本地 HTTP 请求后退出。

路径分析把随机化的 ITT 总效应与经过曝光的干预型中介分解分开：处理后混杂（库存、商家响应）在 g-公式中显式建模而非简单回归控制，自然效应与领券因果效应如实标注为不可识别，缺失曝光不补零、共同支持越界单独披露。方法口径见 `docs/domain.md`，接口参数见 `docs/api.md`。

## 本机运行

使用 Python 3.11，依赖版本固定在 `requirements.txt`。首次准备依赖后，运行和验证不访问公网。

```bash
python3 -m pip install -r requirements.txt
python3 -m reachpath
```

默认资料目录为 `data`，研究备注写入 `state`；入口支持 `--data`、`--state`、`--host` 和 `--port`。研究记录含完整源表副本和内容校验，不会因源文件后续更改自动变成新的研究记录。

## 测试和编译

在项目根目录执行，测试会自行创建临时状态目录并清理，HTTP 测试由系统分配空闲端口。

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q reachpath tests smoke.py
python3 -m build --wheel --no-isolation
python3 smoke.py
```

## 容器运行

镜像固定 Python 基础镜像内容摘要及数值依赖版本，环境准备阶段需要包镜像。验证阶段可完全关闭网络，仅在容器内访问环回 HTTP。

```bash
docker build -t reachpath:local .
docker run --rm --network none reachpath:local python -m unittest discover -s tests -v
docker run --rm --network none reachpath:local python smoke.py
docker run --rm -p 127.0.0.1:8080:8080 reachpath:local
```

完整接口检查通常只需数秒。数值剖面的金额转为浮点仅为探索用途，成本金额对账仍以原始十进制字段为准。
