# run_all_tests.py

import unittest


def main():
    # Discover and run all unittests in the current directory and subdirectories
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='.', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All tests passed successfully.")
    else:
        print(
            f"\n❌ Some tests failed. Failures: {len(result.failures)} Errors: {len(result.errors)}")


if __name__ == "__main__":
    main()
