import unittest
import numpy as np
from env import MarsRover
from pe_td_zero import td_zero_step, evaluate_policy_td_zero

class TestTD(unittest.TestCase):

    def test_td_step(self):
        sample = [2, 0, 0, 1]
        v = np.zeros(5)
        v_new = td_zero_step(v, sample)
        self.assertTrue(np.array_equal(v, v_new))

        sample = [1, 0, 1, 0]
        v = np.zeros(5)
        v_new = td_zero_step(v, sample)
        self.assertFalse(np.array_equal(v, v_new))
        self.assertTrue(v_new[1] == 0.1)

        sample = [1, 0, 1, 0]
        v = np.zeros(5)
        v_new = td_zero_step(v, sample, alpha=0.9)
        self.assertFalse(np.array_equal(v, v_new))
        self.assertTrue(v_new[1] == 0.9)

    def test_td_eval(self):
        pi = np.zeros(5)
        v, i = evaluate_policy_td_zero(pi, rewards=[1, -1, -1, -1, 10])
        self.assertTrue(i > 1)
        self.assertTrue(v[4]==0)
        self.assertTrue(v[3]==0)

        v, i = evaluate_policy_td_zero(pi, rewards=np.zeros(5))
        self.assertTrue(i == 1)

if __name__ == '__main__':
    unittest.main()
