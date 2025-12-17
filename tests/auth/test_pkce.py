"""
Unit tests for PKCE utilities.
"""

import base64
import hashlib
import unittest


class TestPKCE(unittest.TestCase):
    """Tests for PKCE code verifier and challenge generation."""

    def test_generate_code_verifier_length(self):
        """Code verifier should produce a valid Base64 URL-safe string."""
        from litellm.auth.pkce import generate_code_verifier

        verifier = generate_code_verifier()
        # Default 32 bytes -> 43 characters in base64url without padding
        self.assertEqual(len(verifier), 43)
        # Should be URL-safe (no +, /, or =)
        self.assertNotIn("+", verifier)
        self.assertNotIn("/", verifier)
        self.assertNotIn("=", verifier)

    def test_generate_code_verifier_randomness(self):
        """Each call should produce a different verifier."""
        from litellm.auth.pkce import generate_code_verifier

        verifier1 = generate_code_verifier()
        verifier2 = generate_code_verifier()
        self.assertNotEqual(verifier1, verifier2)

    def test_generate_code_challenge_format(self):
        """Code challenge should be SHA-256 hash of verifier in base64url."""
        from litellm.auth.pkce import generate_code_challenge, generate_code_verifier

        verifier = generate_code_verifier()
        challenge = generate_code_challenge(verifier)

        # Verify the challenge manually
        expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
        self.assertEqual(challenge, expected_challenge)

    def test_generate_code_challenge_no_padding(self):
        """Challenge should not have padding characters."""
        from litellm.auth.pkce import generate_code_challenge, generate_code_verifier

        for _ in range(10):
            verifier = generate_code_verifier()
            challenge = generate_code_challenge(verifier)
            self.assertNotIn("=", challenge)

    def test_generate_pkce_pair(self):
        """Pair generation should return valid verifier and challenge."""
        from litellm.auth.pkce import generate_pkce_pair

        verifier, challenge = generate_pkce_pair()
        self.assertIsInstance(verifier, str)
        self.assertIsInstance(challenge, str)
        self.assertEqual(len(verifier), 43)
        self.assertEqual(len(challenge), 43)

        # Verify they're related correctly
        expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
        self.assertEqual(challenge, expected_challenge)

    def test_generate_state(self):
        """State generation should produce random URL-safe strings."""
        from litellm.auth.pkce import generate_state

        state1 = generate_state()
        state2 = generate_state()
        self.assertNotEqual(state1, state2)
        # Default 16 bytes -> 22 characters
        self.assertEqual(len(state1), 22)
        self.assertNotIn("=", state1)


if __name__ == "__main__":
    unittest.main()
