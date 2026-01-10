import os
import unittest
import asyncio

# Dynamic Import Helper
import importlib.util


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


af_path = os.path.join(
    os.getcwd(), "UTILS - Advanced Python - AsyncIO", "async_fetching.py"
)
af_module = load_module_from_path("async_fetching", af_path)


class TestAsyncFetching(unittest.TestCase):

    def test_fetch_returns_dict(self):
        # We need to run the coroutine in the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(af_module.fetch_ticker("TEST"))
        loop.close()

        self.assertEqual(result["symbol"], "TEST")
        self.assertIsInstance(result["price"], float)


if __name__ == "__main__":
    unittest.main()
