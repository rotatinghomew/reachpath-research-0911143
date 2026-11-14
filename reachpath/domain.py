from collections import defaultdict


def arm_counts(catalog):
    arms = {record["customer_id"]: record["arm"] for record in catalog.rows("assignment")}
    counts = defaultdict(lambda: {"users": 0, "purchases": 0})
    for record in catalog.rows("outcomes"):
        group = str(arms[record["customer_id"]])
        counts[group]["users"] += 1
        counts[group]["purchases"] += int(record["purchased"])
    return dict(counts)


def overview(catalog):
    return {"随机分组描述": arm_counts(catalog), "曝光缺失用户数": sum(record["impressions"] is None for record in catalog.rows("mediators"))}
