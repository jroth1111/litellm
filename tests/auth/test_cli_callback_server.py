import threading
import time
import unittest
import urllib.parse

import httpx


class CliCallbackServerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Import lazily so these tests don't depend on full litellm package init.
        from litellm.auth import cli as cli_mod

        self.cli = cli_mod

    def test_ephemeral_port_redirect_uri_returns_actual_port(self):
        httpd, thread, result, done, actual_redirect = self.cli.start_local_callback_server(
            redirect_uri="http://127.0.0.1:0/callback",
            expected_state="s1",
        )
        try:
            parsed = urllib.parse.urlparse(actual_redirect)
            self.assertEqual(parsed.hostname, "127.0.0.1")
            self.assertNotEqual(parsed.port, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_state_mismatch_sets_error_and_unblocks(self):
        httpd, thread, result, done, actual_redirect = self.cli.start_local_callback_server(
            redirect_uri="http://127.0.0.1:0/callback",
            expected_state="expected",
        )
        try:
            url = f"{actual_redirect}?state=wrong&code=abc"
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
            self.assertEqual(resp.status_code, 400)
            self.assertTrue(done.wait(2.0))
            self.assertEqual(result.error, "state_mismatch")
            self.assertIsNone(result.code)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_missing_code_sets_error_and_unblocks(self):
        httpd, thread, result, done, actual_redirect = self.cli.start_local_callback_server(
            redirect_uri="http://127.0.0.1:0/callback",
            expected_state="expected",
        )
        try:
            url = f"{actual_redirect}?state=expected"
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
            self.assertEqual(resp.status_code, 400)
            self.assertTrue(done.wait(2.0))
            self.assertEqual(result.error, "missing_code")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
