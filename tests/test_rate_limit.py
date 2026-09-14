from __future__ import annotations

import unittest

from backend.app.rate_limit import Limiter, client_ip, client_keys


class LimiterTests(unittest.TestCase):
    """The quota in front of the one route that spends money."""

    def setUp(self):
        self.limiter = Limiter(limit=3, window_seconds=60, daily_limit=10)
        self.keys = client_keys("device-a", None)
        self.now = 1_000.0

    def spend(self, count, keys=None, at=None):
        for index in range(count):
            self.limiter.record(keys or self.keys, (at or self.now) + index)

    def test_allows_up_to_the_limit(self):
        for index in range(3):
            self.assertTrue(self.limiter.check(self.keys, self.now + index).allowed)
            self.limiter.record(self.keys, self.now + index)

    def test_refuses_the_one_after(self):
        self.spend(3)

        decision = self.limiter.check(self.keys, self.now + 3)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "burst")
        self.assertGreater(decision.retry_after, 0)

    def test_the_window_slides(self):
        self.spend(3)

        self.assertTrue(self.limiter.check(self.keys, self.now + 61).allowed)

    def test_a_daily_ceiling_sits_above_the_burst_one(self):
        for block in range(4):
            self.spend(3, at=self.now + block * 120)

        decision = self.limiter.check(self.keys, self.now + 600)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "daily")

    def test_a_day_later_the_slate_is_clean(self):
        for block in range(4):
            self.spend(3, at=self.now + block * 120)

        self.assertTrue(self.limiter.check(self.keys, self.now + 86_401 + 360).allowed)

    def test_checking_does_not_count(self):
        """Check and record are separate so a refused request is not charged for
        being refused, which would otherwise extend its own lockout."""
        for _ in range(10):
            self.limiter.check(self.keys, self.now)

        self.assertTrue(self.limiter.check(self.keys, self.now).allowed)

    def test_one_device_does_not_spend_anothers_quota(self):
        self.spend(3)

        self.assertTrue(self.limiter.check(client_keys("device-b", None), self.now + 3).allowed)

    def test_an_address_gets_room_for_a_crowd(self):
        """A village behind one NAT shares an address. Holding that address to
        one farmer's ceiling would cut off everybody in it the moment one of
        them asked a lot of questions."""
        shared = "197.255.0.1"
        for index in range(6):
            self.spend(3, keys=client_keys(f"device-{index}", shared))

        # Eighteen questions through one address, and the next farmer there is
        # still served.
        self.assertTrue(self.limiter.check(client_keys("device-new", shared), self.now + 3).allowed)

    def test_an_address_is_still_a_ceiling(self):
        shared = "197.255.0.1"
        for index in range(20):
            self.spend(3, keys=client_keys(f"device-{index}", shared))

        decision = self.limiter.check(client_keys("device-new", shared), self.now + 3)

        self.assertFalse(decision.allowed)

    def test_memory_is_bounded(self):
        limiter = Limiter(limit=3, window_seconds=60, daily_limit=10, max_keys=50)

        for index in range(500):
            limiter.record(client_keys(f"device-{index}", None), self.now)

        self.assertLessEqual(len(limiter._hits), 50)


class DecisionMessageTests(unittest.TestCase):
    def test_speaks_to_the_farmer_not_the_operator(self):
        limiter = Limiter(limit=1, window_seconds=600, daily_limit=10)
        keys = client_keys("device-a", None)
        limiter.record(keys, 1_000.0)

        message = limiter.check(keys, 1_000.0).message

        self.assertIn("try again", message.lower())
        # No quota vocabulary, no numbers to decode.
        self.assertNotIn("limit", message.lower())
        self.assertNotIn("429", message)


class ClientKeyTests(unittest.TestCase):
    def test_counts_a_request_against_both_identities(self):
        self.assertEqual(client_keys("abc", "1.2.3.4"), ["device:abc", "ip:1.2.3.4"])

    def test_falls_back_when_a_caller_sends_neither(self):
        """A caller with no device id and no resolvable address is exactly the
        shape of a script, so it gets the strictest bucket: a shared one."""
        self.assertEqual(client_keys(None, None), ["anonymous"])

    def test_a_long_device_id_cannot_grow_the_key_space(self):
        key = client_keys("d" * 500, None)[0]

        self.assertLessEqual(len(key), len("device:") + 64)

    def test_reads_the_first_hop_of_a_forwarded_header(self):
        headers = {"x-forwarded-for": "197.255.0.1, 10.0.0.7, 10.0.0.8"}

        self.assertEqual(client_ip(headers, "10.0.0.8"), "197.255.0.1")

    def test_falls_back_to_the_socket_address(self):
        self.assertEqual(client_ip({}, "10.0.0.8"), "10.0.0.8")


if __name__ == "__main__":
    unittest.main()
