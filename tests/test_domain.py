import unittest
from reachpath.catalog import Catalog
from reachpath.domain import arm_counts


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog("data")

    def test_all_assigned_users_enter_arm_description(self):
        self.assertEqual(sum(record["users"] for record in arm_counts(self.catalog).values()), len(self.catalog.rows("assignment")))
        self.assertTrue(any(record["impressions"] is None for record in self.catalog.rows("mediators")))


    def test_claim_status_does_not_reassign_arm(self):
        catalog = FakeCatalog({"assignment": [{"customer_id": "A", "arm": 1}],
                               "outcomes": [{"customer_id": "A", "purchased": False, "coupon_claimed": False}]})
        self.assertEqual(arm_counts(catalog), {"1": {"users": 1, "purchases": 0}})

    def test_control_purchases_stay_in_control(self):
        catalog = FakeCatalog({"assignment": [{"customer_id": "A", "arm": 0}],
                               "outcomes": [{"customer_id": "A", "purchased": True}]})
        self.assertEqual(arm_counts(catalog), {"0": {"users": 1, "purchases": 1}})

    def test_empty_assignments(self):
        self.assertEqual(arm_counts(FakeCatalog({"assignment": [], "outcomes": []})), {})

    def test_baseline_precedes_assignment(self):
        assignments = {record["customer_id"]: record for record in self.catalog.rows("assignment")}
        for record in self.catalog.rows("baseline"):
            self.assertLess(record["available_at"], assignments[record["customer_id"]]["assigned_at"])


class FakeCatalog:
    def __init__(self, tables):
        self.tables = tables

    def rows(self, name):
        return self.tables[name]
