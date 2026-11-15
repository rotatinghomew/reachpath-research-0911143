"""随机返券实验的路径（中介）分析。

时序 DAG（每个用户内部时间截点严格有序）：

    X(分组前特征) -> Z(随机返券资格) -> R(库存/商家响应) -> M(曝光) -> Y(成交)
          |              |     |              |            |
          |              |     +------------> M            |
          |              +--------------------> Y          |
          +------------------------------------> Y <------+
                         Z -> C(实际领券)，C 的领券身份不用于因果调整

R 在分组之后、曝光之前记录，既受 Z 影响，又同时影响 M 与 Y，是处理后混杂
（treatment-induced confounder）。因此自然直接/间接效应（NDE/NIE）在现有数据下
不可非参数识别；本模块选择与时序相容的干预型中介目标（interventional analog,
VanderWeele et al. 2014），通过 g-公式识别 IDE/IIE，并明确列出所需额外假设。

随机化本身支持的意向处理（ITT）总效应始终单独报告，不与任何模型估计混同。
"""

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

from .cases import canonical


def _parse_time(value):
    return datetime.fromisoformat(value)

DAG = {
    "nodes": [
        {"id": "X", "label": "分组前特征", "source": "baseline.prior_orders / baseline.region",
         "phase": "分组前", "randomized": False},
        {"id": "Z", "label": "随机返券资格（arm）", "source": "assignment.arm",
         "phase": "随机分组", "randomized": True},
        {"id": "R", "label": "库存与商家响应", "source": "response.stock / response.merchant_response",
         "phase": "分组后、曝光前", "randomized": False},
        {"id": "M", "label": "曝光次数", "source": "mediators.impressions",
         "phase": "中介（曝光窗口）", "randomized": False,
         "missing_semantics": "空值为整段统计丢失，不是零曝光，禁止补零"},
        {"id": "C", "label": "实际领券", "source": "outcomes.coupon_claimed",
         "phase": "分组后行为", "randomized": False,
         "note": "领券是处理后行为，不进入因果调整，领券者对比不是随机对比"},
        {"id": "Y", "label": "成交", "source": "outcomes.purchased",
         "phase": "结果（成交窗口）", "randomized": False},
    ],
    "edges": [
        {"from": "X", "to": "Z", "relation": "分层随机化的分层来源"},
        {"from": "X", "to": "R", "relation": "基线顾客特征影响供给与商家响应"},
        {"from": "X", "to": "M", "relation": "基线特征影响曝光"},
        {"from": "X", "to": "Y", "relation": "基线特征影响成交"},
        {"from": "Z", "to": "R", "relation": "活动改变库存与商家响应（处理后混杂来源）"},
        {"from": "Z", "to": "M", "relation": "活动资格影响曝光"},
        {"from": "Z", "to": "Y", "relation": "资格对成交的直接路径"},
        {"from": "Z", "to": "C", "relation": "资格决定领券机会（对照组无券可领）"},
        {"from": "R", "to": "M", "relation": "库存与商家响应左右曝光"},
        {"from": "R", "to": "Y", "relation": "库存与商家响应左右成交"},
        {"from": "M", "to": "Y", "relation": "曝光影响成交"},
    ],
    "non_edges": [
        "C -> Y 不作为识别用结构边：领券与曝光、供给同属处理后行为，无法用现有随机化分离",
        "R 不是分组前协变量，不能按普通协变量在回归中控制后宣称直接效应",
    ],
}

TIMELINE_RULE = [
    "baseline.available_at", "assignment.assigned_at", "response.occurred_at",
    "mediators.window_start", "mediators.window_end",
    "outcomes.window_start", "outcomes.window_end",
]

INTERVENTIONAL_ASSUMPTIONS = [
    "一致性：观测到的 (Z,R,M,Y) 等于对应干预下的潜在结果",
    "可交换性：Z 为随机分组，Z 独立于各潜在结果（可按 X 分层调整）",
    "无干扰（SUTVA）：一名用户的潜在结果不依赖他人的分组与曝光",
    "处理后混杂 R 被完整观测；随机中介干预分布 G_z' 用 arm=z' 训练折的经验 (R,M) 联合分布表示，保留 R-M 相关且不假设 R 关于 X 的参数模型",
    "结果模型 q(Y|Z,R,M,X) 为按 arm 拟合的线性概率模型，设定正确；M 对 Y 的线性外推在共同支持内有效，越界区间依赖外推",
    "曝光缺失为 MAR：M 缺失独立于潜在 M、Y，给定 (Z,X,R)；经验中介池只含完整案例，不做零填补",
    "正定性：组间缺乏共同支持的区间没有观测数据，报告越界单元数而非隐式补值",
]

NATURAL_EFFECT_NOTE = (
    "自然直接/间接效应（NDE/NIE）需要跨世界中介分布 M(z) 与结果模型的联合分布。"
    "R 受 Z 影响且同时混杂 M->Y，属于处理后混杂；对其条件化会阻断 Z->R->Y 路径，"
    "不对其条件化则 M->Y 存在混杂，跨世界分布在现有数据中没有任何观测对应，"
    "因此 NDE/NIE 在现有数据下不可非参数识别，只能在敏感性分析中做假设性思想实验。"
)

IV_NOTE = (
    "Z 对 C 是有效工具（对照组无人领券，单调性由设计保证），但 Wald 估计仅在排他性约束"
    "（Z 只通过领券影响 Y）成立时才是领券的 CACE；本 DAG 中 Z 还经 R、M 影响 Y，"
    "排他性约束结构上不成立，故领券的因果效应不可识别。处理组内领券与否的成交率差异"
    "是观察性对比，不能冒充随机实验结论。"
)

NAIVE_NOTE = (
    "在 pooled 回归中直接控制 R、M 后读取 Z 的系数，混合了阻断处理后路径与碰撞调整，"
    "不是因果直接效应，仅作为反例展示，不参与任何路径分解。"
)


def _index(catalog, name):
    table = {}
    for record in catalog.rows(name):
        customer = record["customer_id"]
        if customer in table:
            raise ValueError(f"{name} 中同一用户出现多次：{customer}")
        table[customer] = record
    return table


def assemble(catalog):
    """按 customer_id 一对一合表，并核对每个用户的事件截点次序。"""
    assignment = _index(catalog, "assignment")
    baseline = _index(catalog, "baseline")
    response = _index(catalog, "response")
    mediators = _index(catalog, "mediators")
    outcomes = _index(catalog, "outcomes")
    keys = set(assignment)
    for name, table in (("baseline", baseline), ("response", response),
                        ("mediators", mediators), ("outcomes", outcomes)):
        if set(table) != keys:
            raise ValueError(f"{name} 的用户集合与 assignment 不一致")
    rows = []
    timing_violations = []
    spans = {phase: [None, None] for phase in
             ("baseline", "assignment", "response", "impression_window", "outcome_window")}
    for customer in sorted(keys):
        a, b = assignment[customer], baseline[customer]
        r, m, o = response[customer], mediators[customer], outcomes[customer]
        ordered = [_parse_time(value) for value in (
            b["available_at"], a["assigned_at"], r["occurred_at"],
            m["window_start"], m["window_end"], o["window_start"], o["window_end"])]
        # 窗口无重叠允许曝光结束与成交开始在同一时刻相接，故窗口边界用严格不等号，
        # 事件截点（基线/分组/响应/窗口起点）之间仍要求严格先后。
        strict_gaps = (0, 1, 2, 3, 5)
        if any(ordered[i] > ordered[i + 1] if i not in strict_gaps
               else ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            timing_violations.append(customer)
        for phase, start, end in (
                ("baseline", b["available_at"], b["available_at"]),
                ("assignment", a["assigned_at"], a["assigned_at"]),
                ("response", r["occurred_at"], r["occurred_at"]),
                ("impression_window", m["window_start"], m["window_end"]),
                ("outcome_window", o["window_start"], o["window_end"])):
            low, high = spans[phase]
            spans[phase] = [start if low is None else min(low, start),
                            end if high is None else max(high, end)]
        rows.append({
            "customer_id": customer, "z": int(a["arm"]), "stratum": a["stratum"],
            "probability": float(a["probability"]),
            "prior_orders": int(b["prior_orders"]), "region": b["region"],
            "stock": int(r["stock"]), "merch": float(r["merchant_response"]),
            "m": None if m["impressions"] is None else int(m["impressions"]),
            "y": bool(o["purchased"]), "claim": bool(o["coupon_claimed"]),
        })
    timeline = {
        "rule": TIMELINE_RULE,
        "per_user_ordering_violations": len(timing_violations),
        "violation_customers": timing_violations[:20],
        "observed_spans": {k: {"earliest": v[0], "latest": v[1]} for k, v in spans.items()},
    }
    return rows, timeline


def _levels(rows):
    return {
        "strata": sorted({row["stratum"] for row in rows}),
        "regions": sorted({row["region"] for row in rows}),
    }


def _fit_centers(train_rows):
    complete = [row for row in train_rows if row["m"] is not None]
    return {
        "prior": float(np.mean([row["prior_orders"] for row in train_rows])),
        "stock": float(np.mean([row["stock"] for row in train_rows])),
        "merch": float(np.mean([row["merch"] for row in train_rows])),
        "m": float(np.mean([row["m"] for row in complete])) if complete else 0.0,
    }


def _design(rows, centers, levels, include_z, include_r=False, include_m=False, force_z=None):
    """构建设计矩阵。数值列的中心化常数只允许来自训练折。

    列布局：intercept、(arm)、prior_orders、stratum 哑变量、region 哑变量、
    (stock、merchant_response)、(impressions)，括号中的列按开关加入。
    """
    strata_dummies = levels["strata"][1:]
    region_dummies = levels["regions"][1:]
    width = 1 + int(include_z) + 1 + len(strata_dummies) + len(region_dummies)
    if include_r:
        width += 2
    if include_m:
        width += 1
    matrix = np.zeros((len(rows), width))
    intercept_col = 0
    arm_col = 1
    prior_col = int(include_z) + 1
    stratum_base = prior_col + 1
    region_base = stratum_base + len(strata_dummies)
    extra_base = region_base + len(region_dummies)
    strata_map = {value: stratum_base + index for index, value in enumerate(strata_dummies)}
    region_map = {value: region_base + index for index, value in enumerate(region_dummies)}
    for i, row in enumerate(rows):
        z = row["z"] if force_z is None else force_z
        matrix[i, intercept_col] = 1.0
        if include_z:
            matrix[i, arm_col] = float(z)
        matrix[i, prior_col] = row["prior_orders"] - centers["prior"]
        if row["stratum"] in strata_map:
            matrix[i, strata_map[row["stratum"]]] = 1.0
        if row["region"] in region_map:
            matrix[i, region_map[row["region"]]] = 1.0
        cursor = extra_base
        if include_r:
            matrix[i, cursor] = row["stock"] - centers["stock"]
            matrix[i, cursor + 1] = row["merch"] - centers["merch"]
            cursor += 2
        if include_m:
            matrix[i, cursor] = (row["m"] if row["m"] is not None else 0.0) - centers["m"]
    return matrix


def _ols(x, y):
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _irls(x, y, link, iterations=25):
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        eta = x @ beta
        if link == "logit":
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            weight = np.maximum(p * (1 - p), 1e-8)
            working = eta + (y - p) / weight
        else:
            p = norm.cdf(eta)
            density = norm.pdf(eta)
            weight = np.maximum(density ** 2 / np.maximum(p * (1 - p), 1e-8), 1e-8)
            working = eta + (y - p) / np.maximum(density, 1e-8)
        beta = _ols(x * np.sqrt(weight)[:, None], working * np.sqrt(weight))
    return beta


def _make_folds(rows, k, rng):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["z"], row["stratum"]), []).append(index)
    folds = [[] for _ in range(k)]
    for indices in groups.values():
        shuffled = np.asarray(indices)[rng.permutation(len(indices))]
        for position, index in enumerate(shuffled):
            folds[position % k].append(int(index))
    return folds


def _fit_arm_models(train_rows, arm, centers, levels):
    """训练折内、按 arm 拟合结果模型 q(Y|Z=arm,R,M,X)。"""
    arm_rows = [row for row in train_rows if row["z"] == arm]
    complete = [row for row in arm_rows if row["m"] is not None]
    y_x = _design(complete, centers, levels, include_z=False, include_r=True, include_m=True)
    y_y = np.asarray([float(row["y"]) for row in complete])
    beta_y = _ols(y_x, y_y)
    return {"beta_y": beta_y}


def _empirical_mediator_pool(train_rows, arm):
    """训练折内 arm 组的经验 (stock, merch, impressions) 联合样本（完整案例）。

    随机中介干预 G_z' 直接从该臂观测到的 R-M 联合分布有放回抽样，
    不假设 R 关于 X 的参数模型，对弱 X->R 首阶段更稳健，
    并天然保留 R 与 M 的相关结构。缺失 M 的单元不进入该池，绝不补零。
    """
    pool = [(row["stock"], row["merch"], row["m"]) for row in train_rows
            if row["z"] == arm and row["m"] is not None]
    if len(pool) < 2:
        raise ValueError(f"arm {arm} 的完整曝光样本不足，无法构造经验中介干预分布")
    return np.asarray(pool, dtype=float)


def _simulate_mean(held_out, models_by_arm, pools, z, zp, centers, levels, draws, rng):
    """Monte Carlo 估计 E[Y(z, G_{z'})] 的均值。

    R、M 从 arm=z' 训练折的经验联合分布抽取（随机中介干预分布 G_{z'}），
    Y 用 arm=z 的交叉拟合结果模型 q(Y|Z=z,R,M,X) 预测。
    """
    base = _design(held_out, centers, levels, include_z=False)
    n = len(held_out)
    k = base.shape[1]
    pool = pools[zp]
    total = 0.0
    for _ in range(draws):
        pick = rng.integers(0, len(pool), size=n)
        stock = pool[pick, 0]
        merch = pool[pick, 1]
        m_draw = pool[pick, 2]
        y_design = np.zeros((n, k + 3))
        y_design[:, :k] = base
        y_design[:, k] = stock - centers["stock"]
        y_design[:, k + 1] = merch - centers["merch"]
        y_design[:, k + 2] = m_draw - centers["m"]
        total += float(np.mean(y_design @ models_by_arm[z]["beta_y"]))
    return total / draws


def gformula(rows, levels, rng, folds=5, mc_draws=8):
    """交叉拟合的干预型 g-公式；返回四个 (z, z') 联合干预均值。"""
    fold_ids = _make_folds(rows, folds, rng)
    assigned = np.empty(len(rows), dtype=int)
    for fold_id, indices in enumerate(fold_ids):
        assigned[indices] = fold_id
    means = {(z, zp): [] for z in (0, 1) for zp in (0, 1)}
    adjusted_itt = []
    for fold_id, indices in enumerate(fold_ids):
        held = [rows[i] for i in indices]
        train = [row for i, row in enumerate(rows) if assigned[i] != fold_id]
        centers = _fit_centers(train)
        models_by_arm = {arm: _fit_arm_models(train, arm, centers, levels) for arm in (0, 1)}
        pools = {arm: _empirical_mediator_pool(train, arm) for arm in (0, 1)}
        for z in (0, 1):
            for zp in (0, 1):
                means[(z, zp)].append(_simulate_mean(
                    held, models_by_arm, pools, z, zp,
                    centers, levels, mc_draws, rng))
        # 交叉拟合的分层调整 ITT：模型只在训练折拟合，在留出折反事实预测
        train_x = _design(train, centers, levels, include_z=True)
        beta_itt = _ols(train_x, np.asarray([float(row["y"]) for row in train]))
        for force_z in (0, 1):
            held_x = _design(held, centers, levels, include_z=True, force_z=force_z)
            adjusted_itt.append((force_z, float(np.mean(held_x @ beta_itt))))
    m = {key: float(np.mean(values)) for key, values in means.items()}
    p0 = np.mean([value for z, value in adjusted_itt if z == 0])
    p1 = np.mean([value for z, value in adjusted_itt if z == 1])
    return m, float(p1 - p0)


def _ci(replicates):
    array = np.asarray(replicates, dtype=float)
    return {"estimate": float(np.mean(array)),
            "se": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]}


def _raw_itt(rows, field, complete_only=False):
    groups = {0: [], 1: []}
    for row in rows:
        if complete_only and row["m"] is None:
            continue
        groups[row["z"]].append(row[field] if row[field] is not None else np.nan)
    stats = {}
    for arm, values in groups.items():
        array = np.asarray(values, dtype=float)
        stats[arm] = {"mean": float(np.mean(array)), "var": float(np.var(array, ddof=1)),
                      "n": int(array.size)}
    diff = stats[1]["mean"] - stats[0]["mean"]
    se = math.sqrt(stats[1]["var"] / stats[1]["n"] + stats[0]["var"] / stats[0]["n"])
    return diff, se, stats


def _positivity(rows):
    support = {}
    for arm in (0, 1):
        values = [row["m"] for row in rows if row["z"] == arm and row["m"] is not None]
        support[f"arm_{arm}"] = {"min": int(min(values)), "max": int(max(values)),
                                 "observed": len(values)}
    treated_outside = [row["customer_id"] for row in rows
                       if row["z"] == 1 and row["m"] is not None
                       and (row["m"] < support["arm_0"]["min"] or row["m"] > support["arm_0"]["max"])]
    control_outside = [row["customer_id"] for row in rows
                       if row["z"] == 0 and row["m"] is not None
                       and (row["m"] < support["arm_1"]["min"] or row["m"] > support["arm_1"]["max"])]
    return {
        "impression_support": support,
        "treated_units_outside_control_support": len(treated_outside),
        "control_units_outside_treated_support": len(control_outside),
        "example_treated_outside": treated_outside[:10],
        "example_control_outside": control_outside[:10],
        "zero_imputation_used": False,
        "note": "越界单元的反事实曝光没有观测数据支持，g-公式在此依赖参数外推；缺失曝光保持缺失，不补零。",
    }


def _sensitivity(rows, levels, rng, grid_size=19, sims=200):
    """Imai 风格 rho 敏感性（顺序可忽略性下的思想实验，probit 结果模型）。"""
    complete = [row for row in rows if row["m"] is not None]
    centers = _fit_centers(complete)
    x_base = _design(complete, centers, levels, include_z=True)
    m_values = np.asarray([row["m"] for row in complete], dtype=float)
    alpha = _ols(x_base, m_values)
    sigma_m = float(np.std(m_values - x_base @ alpha))
    y_values = np.asarray([float(row["y"]) for row in complete])
    x_outcome = np.column_stack([x_base, m_values - centers["m"]])
    beta = _irls(x_outcome, y_values, link="probit")
    x0 = x_base.copy()
    x0[:, 1] = 0.0
    base_eta = x0 @ beta[:x_base.shape[1]]
    alpha_z = alpha[1]
    beta_z, beta_m = beta[1], beta[-1]
    eps = rng.normal(0.0, sigma_m, size=(sims, len(complete)))
    independent = rng.normal(0.0, 1.0, size=(sims, len(complete)))
    m0 = (x0 @ alpha) + eps
    m1 = m0 + alpha_z
    grid = np.linspace(-0.9, 0.9, grid_size)
    table = []
    for rho in grid:
        nu = rho * eps / sigma_m + math.sqrt(1 - rho ** 2) * independent
        p00 = np.mean(norm.cdf(base_eta + beta_m * (m0 - centers["m"]) + nu))
        p01 = np.mean(norm.cdf(base_eta + beta_m * (m1 - centers["m"]) + nu))
        p10 = np.mean(norm.cdf(base_eta + beta_z + beta_m * (m0 - centers["m"]) + nu))
        p11 = np.mean(norm.cdf(base_eta + beta_z + beta_m * (m1 - centers["m"]) + nu))
        acme0 = p01 - p00
        acme1 = p11 - p10
        ade0 = p10 - p00
        ade1 = p11 - p01
        table.append({"rho": float(rho), "acme_at_z0": float(acme0), "acme_at_z1": float(acme1),
                      "acme_average": float((acme0 + acme1) / 2),
                      "ade_at_z0": float(ade0), "ade_at_z1": float(ade1)})
    average = np.asarray([row["acme_average"] for row in table])
    crossing = None
    sign_changes = np.where(np.sign(average[:-1]) != np.sign(average[1:]))[0]
    if sign_changes.size:
        i = int(sign_changes[0])
        x0v, x1v = grid[i], grid[i + 1]
        y0v, y1v = average[i], average[i + 1]
        crossing = float(x0v - y0v * (x1v - x0v) / (y1v - y0v))
    return {
        "identified": False,
        "model": "中介线性模型 + 成交 probit 潜变量模型（仅含分组前 X，刻意不纳入处理后混杂 R），误差相关系数 rho 不可识别",
        "rho_grid": [float(value) for value in grid],
        "rho_at_acme_zero": crossing,
        "rho_crossing_note": (
            "在给定 rho 网格内 ACME 均未过零：要推翻正向中介结论需要更强的未测混杂假设；"
            "rho_at_acme_zero 为 null 表示网格 [-0.9, 0.9] 内不存在翻转点"
        ) if crossing is None else "ACME 过零所需的假设相关系数",
        "grid": table,
        "assumptions": [
            "顺序可忽略性（无未测 M-Y 混杂，即假设 R 与 X 已充分）——本数据中该假设不可验证",
            "线性/probit 参数设定、误差联合正态",
            "rho 是 M、Y 未测混杂相关的假设参数，不是估计量；rho=0 对应顺序可忽略性",
        ],
        "note": NATURAL_EFFECT_NOTE,
    }


def _naive_regression(rows, levels):
    complete = [row for row in rows if row["m"] is not None]
    centers = _fit_centers(complete)
    x = _design(complete, centers, levels, include_z=True, include_r=True, include_m=True)
    beta = _ols(x, np.asarray([float(row["y"]) for row in complete]))
    names = ["intercept", "arm", "prior_orders"]
    names += [f"stratum:{level}" for level in levels["strata"][1:]]
    names += [f"region:{level}" for level in levels["regions"][1:]]
    names += ["stock", "merchant_response", "impressions"]
    if len(names) != x.shape[1]:
        raise RuntimeError("朴素回归列名与设计矩阵不一致")
    return {"coefficients": {name: float(value) for name, value in zip(names, beta)},
            "coef_arm": float(beta[1]), "coef_impressions": float(beta[-1]),
            "warning": NAIVE_NOTE}


def _missingness(rows, rng):
    rates = {}
    for arm in (0, 1):
        arm_rows = [row for row in rows if row["z"] == arm]
        missing = sum(row["m"] is None for row in arm_rows)
        rates[f"arm_{arm}"] = {"missing": missing, "n": len(arm_rows),
                               "rate": missing / len(arm_rows)}
    x = np.ones((len(rows), 2))
    y = np.zeros(len(rows))
    for i, row in enumerate(rows):
        x[i, 1] = row["z"]
        y[i] = float(row["m"] is None)
    beta = _irls(x, y, link="logit")
    return {"rates_by_arm": rates,
            "logistic_intercept": float(beta[0]), "logistic_coef_arm": float(beta[1]),
            "mar_required": INTERVENTIONAL_ASSUMPTIONS[4],
            "note": "缺失率组间不同本身不否定 MAR，但意味着完整案例推断依赖该假设；任何情形下都不得把缺失当零曝光。"}


def _bootstrap(rows, levels, seed, boot_reps, folds, mc_draws):
    base_rng = np.random.default_rng(seed)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["z"], row["stratum"]), []).append(index)
    records = []
    for rep in range(boot_reps):
        rng = np.random.default_rng(base_rng.integers(0, 2 ** 31))
        sampled = []
        for indices in groups.values():
            picks = rng.integers(0, len(indices), size=len(indices))
            sampled.extend(rows[indices[i]] for i in picks)
        means, adjusted = gformula(sampled, levels, rng, folds=folds, mc_draws=mc_draws)
        itt_m, _, _ = _raw_itt(sampled, "m", complete_only=True)
        records.append({
            "te": means[(1, 1)] - means[(0, 0)],
            "ide_z0": means[(1, 0)] - means[(0, 0)],
            "iie_z1": means[(1, 1)] - means[(1, 0)],
            "ide_z1": means[(1, 1)] - means[(0, 1)],
            "iie_z0": means[(0, 1)] - means[(0, 0)],
            "adjusted_itt": adjusted,
            "assignment_on_m": itt_m,
        })
    return {key: _ci([record[key] for record in records]) for key in records[0]}


def analyze(catalog, seed=20261115, boot_reps=200, folds=5, mc_draws=8, run_sensitivity=True):
    rows, timeline = assemble(catalog)
    levels = _levels(rows)
    rng = np.random.default_rng(seed)
    n_by_arm = {f"arm_{arm}": sum(row["z"] == arm for row in rows) for arm in (0, 1)}

    itt_diff, itt_se, itt_stats = _raw_itt(rows, "y")
    arm_rates = {f"arm_{arm}": {"purchase_rate": itt_stats[arm]["mean"], "n": itt_stats[arm]["n"]}
                 for arm in (0, 1)}
    m_diff, m_se, m_stats = _raw_itt(rows, "m", complete_only=True)

    means, adjusted_itt = gformula(rows, levels, rng, folds=folds, mc_draws=mc_draws)
    boot = _bootstrap(rows, levels, seed + 1, boot_reps, folds, mc_draws)

    g_means = {f"Y({z},G{zp})": means[(z, zp)] for z in (0, 1) for zp in (0, 1)}
    te = means[(1, 1)] - means[(0, 0)]
    ide_z0 = means[(1, 0)] - means[(0, 0)]
    iie_z1 = means[(1, 1)] - means[(1, 0)]
    ide_z1 = means[(1, 1)] - means[(0, 1)]
    iie_z0 = means[(0, 1)] - means[(0, 0)]
    calibration = {
        "natural_arm_predicted_vs_observed": {
            "control": {"predicted_Y(0,G0)": means[(0, 0)],
                        "observed_purchase_rate": itt_stats[0]["mean"]},
            "treated": {"predicted_Y(1,G1)": means[(1, 1)],
                        "observed_purchase_rate": itt_stats[1]["mean"]},
        },
        "gformula_total_effect_vs_itt": {"gformula_te": te, "randomized_itt": itt_diff,
                                          "difference": te - itt_diff},
        "interpretation": "自然臂预测与观测成交率、g-公式 TE 与随机化 ITT 应相互接近；明显背离提示模型设定或外推问题。",
    }

    claim_rates = {}
    for arm in (0, 1):
        arm_rows = [row for row in rows if row["z"] == arm]
        claims = sum(row["claim"] for row in arm_rows)
        claim_rates[f"arm_{arm}"] = {"claim_rate": claims / len(arm_rows), "claims": claims,
                                     "n": len(arm_rows)}
    treated = [row for row in rows if row["z"] == 1]
    claimed = [row for row in treated if row["claim"]]
    unclaimed = [row for row in treated if not row["claim"]]
    claim_contrast = (float(np.mean([float(row["y"]) for row in claimed])) -
                      float(np.mean([float(row["y"]) for row in unclaimed])))
    wald = itt_diff / claim_rates["arm_1"]["claim_rate"]

    report = {
        "source_fingerprint": getattr(catalog, "fingerprint", "unversioned"),
        "parameters": {"seed": seed, "boot_reps": boot_reps, "folds": folds,
                       "mc_draws": mc_draws, "sensitivity": run_sensitivity},
        "dag": DAG,
        "timeline": timeline,
        "sample": {"n": len(rows), "by_arm": n_by_arm,
                   "complete_impressions": sum(row["m"] is not None for row in rows),
                   "missing_impressions": sum(row["m"] is None for row in rows)},
        "randomization_estimands": {
            "itt_purchase": {
                "estimate": itt_diff,
                "se_neyman": itt_se,
                "ci95": [itt_diff - 1.96 * itt_se, itt_diff + 1.96 * itt_se],
                "arm_rates": arm_rates,
                "estimand": "意向处理总效应 ITT，仅依赖随机化，无模型假设",
            },
            "itt_adjusted_baseline": {
                **boot["adjusted_itt"],
                "estimand": "按分组前特征（prior_orders/stratum/region）交叉拟合调整的 ITT",
                "cross_fitted": True,
            },
            "assignment_on_impressions": {
                "estimate": m_diff,
                "se_welch": m_se,
                "ci95": [m_diff - 1.96 * m_se, m_diff + 1.96 * m_se],
                "arm_stats": {f"arm_{arm}": {"mean": m_stats[arm]["mean"], "n": m_stats[arm]["n"]}
                              for arm in (0, 1)},
                "complete_cases_only": True,
                "estimand": "Z 对曝光次数的 ITT（缺失曝光不参与、不补零）",
            },
            "impression_missingness": _missingness(rows, rng),
        },
        "interventional_mediation": {
            "estimand": "干预型直接效应 IDE 与干预型间接效应 IIE（g-公式，允许处理后混杂 R）",
            "assumptions": INTERVENTIONAL_ASSUMPTIONS,
            "cross_fitted": True,
            "g_formula_means": g_means,
            "total_effect": {**boot["te"], "point_check": te},
            "ide_reference_z0": {**boot["ide_z0"], "point_check": ide_z0},
            "iie_reference_z1": {**boot["iie_z1"], "point_check": iie_z1},
            "ide_reference_z1": {**boot["ide_z1"], "point_check": ide_z1},
            "iie_reference_z0": {**boot["iie_z0"], "point_check": iie_z0},
            "decomposition_check": {
                "ide_z0_plus_iie_z1_minus_te": ide_z0 + iie_z1 - te,
                "ide_z1_plus_iie_z0_minus_te": ide_z1 + iie_z0 - te,
            },
            "calibration": calibration,
        },
        "natural_mediation": {
            "identified": False,
            "reason": NATURAL_EFFECT_NOTE,
            "sensitivity": _sensitivity(rows, levels, rng) if run_sensitivity else None,
        },
        "positivity": _positivity(rows),
        "naive_regression_control": _naive_regression(rows, levels),
        "coupon_claim": {
            "claim_rates_by_arm": claim_rates,
            "observed_within_treated_claim_contrast": claim_contrast,
            "contrast_warning": "处理组内领券者与未领券者的观察性成交率差异，不是随机对比",
            "wald_iv_number": wald,
            "identification_warning": IV_NOTE,
        },
        "identification": [
            {"estimand": "ITT 总效应（Z->Y）", "status": "已识别",
             "extra_assumptions": [], "basis": "随机分组本身"},
            {"estimand": "Z 对曝光的 ITT（Z->M）", "status": "已识别（完整案例）",
             "extra_assumptions": ["完整案例上的随机化保持；缺失机制单独报告"],
             "basis": "随机分组本身"},
            {"estimand": "干预型直接/间接效应 IDE/IIE", "status": "依赖额外假设",
             "extra_assumptions": INTERVENTIONAL_ASSUMPTIONS[1:],
             "basis": "g-公式 + 交叉拟合参数模型"},
            {"estimand": "自然直接/间接效应 NDE/NIE", "status": "现有数据不可识别",
             "extra_assumptions": ["需不可验证的跨世界独立假设；仅提供 rho 敏感性"],
             "basis": "处理后混杂 R 阻断非参数识别"},
            {"estimand": "实际领券的因果效应 / CACE", "status": "现有数据不可识别",
             "extra_assumptions": ["排他性约束在本 DAG 中结构上不成立"],
             "basis": "Z 经 R、M 另有路径影响 Y"},
        ],
    }
    return _finite(report)


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("分析结果出现非有限数值")
        return round(value, 6)
    return value


class AnalysisStore:
    """不可变分析记录：完整报告 + 内容校验，写法与 CaseStore 一致。"""

    def __init__(self, directory):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, report, label):
        payload = {
            "id": uuid.uuid4().hex,
            "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "report": report,
        }
        envelope = {"content": payload,
                    "sha256": hashlib.sha256(canonical(payload)).hexdigest()}
        with (self.directory / (payload["id"] + ".json")).open("xb") as target:
            target.write(canonical(envelope))
        return payload

    def read(self, identifier):
        if not identifier or not all(ch in "0123456789abcdef" for ch in identifier) or len(identifier) != 32:
            raise KeyError(identifier)
        try:
            envelope = json.loads((self.directory / (identifier + ".json")).read_bytes())
        except FileNotFoundError as error:
            raise KeyError(identifier) from error
        content = envelope["content"]
        if hashlib.sha256(canonical(content)).hexdigest() != envelope["sha256"]:
            raise ValueError("分析记录完整性校验失败")
        return content

    def list(self):
        return [{"id": record["id"], "label": record["label"],
                 "created_at": record["created_at"],
                 "source_fingerprint": record["report"]["source_fingerprint"]}
                for record in (self.read(path.stem) for path in sorted(self.directory.glob("*.json")))]
