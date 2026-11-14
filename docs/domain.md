# 随机返券活动中曝光中介路径资料约定

一位用户仅有一次分组，分析总体以所有随机用户为基础。处理后库存和商家响应在曝光前记录，可能受到返券活动影响。曝光与成交窗口无重叠，时间戳均有时区；缺失曝光不可编码为零。随机资格本身可支持意向处理总效应，不自动赋予曝光或领券随机性。现有证据没有声称所有中介混杂均被观测。

## 数据表

### `assignment`

用户独立随机分组一次，arm 为返券资格，不是实际领券；probability 是处理组概率。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `customer_id` | `text` |
| `arm` | `int` |
| `stratum` | `text` |
| `probability` | `number` |
| `assigned_at` | `time` |

### `baseline`

分组前可用特征，不含活动后行为。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `customer_id` | `text` |
| `prior_orders` | `int` |
| `region` | `text` |
| `available_at` | `time` |

### `response`

活动开始后的供给与商家响应，早于曝光窗口，不属于随机分组前协变量。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `customer_id` | `text` |
| `stock` | `int` |
| `merchant_response` | `number` |
| `occurred_at` | `time` |

### `mediators`

曝光窗口计数；空值是整段统计丢失，不是零曝光。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `customer_id` | `text` |
| `impressions` | `int?` |
| `window_start` | `time` |
| `window_end` | `time` |

### `outcomes`

曝光后成交与是否实际领券。随机资格不强迫领券，也不强迫成交。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `customer_id` | `text` |
| `purchased` | `bool` |
| `coupon_claimed` | `bool` |
| `window_start` | `time` |
| `window_end` | `time` |

## 文件和数值语义

空单元格只在带问号的类型中表示缺失；零和空值不同。金额由十进制定点字符串保存，浮点数仅用于探索性统计。所有数据均为本地生成的虚构经营记录，不对应真实公司、客户或住址。数值剖面只反映观察数据，不能代替总体推断。

研究备注的参数为业务自由对象，不代表已执行了任何统计模型。原始表与已有视图可单独查询。
