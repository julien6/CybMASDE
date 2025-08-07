import unittest

from .test_label_manager import test_label_manager
from .test_organizational_model import test_organizational_model
from .test_organizational_specification_logic import test_organizational_specification_logic_model
from .test_trajectory_function import test_trajectory_function
from .test_trajectory_pattern import test_trajectory_pattern
from .test_global import test_global

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_label_manager))
    suite.addTest(unittest.makeSuite(test_organizational_model))
    suite.addTest(unittest.makeSuite(test_organizational_specification_logic_model))
    suite.addTest(unittest.makeSuite(test_trajectory_function))
    suite.addTest(unittest.makeSuite(test_trajectory_pattern))
    suite.addTest(unittest.makeSuite(test_global))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
