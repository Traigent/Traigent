import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import check_pr_ticket_security as c  # noqa: E402

OKS = [
    "Ticket: TRAIG-123\nSecurity note: none (docs only)",
    "Ticket: Fixes #45\nSecurity note: adds authz check on /x",
    "Ticket: #45\nSecurity note: touches no auth paths",
    "Ticket: https://github.com/Traigent/TraigentBackend/issues/7\nSecurity note: none (refactor)",
    "Ticket: none (typo)\nSecurity note: none (typo)",
    "  ticket: AGB-9  \n  security note: none (x)  ",
    "## T\nTicket: Closes Traigent/traigent-iac#12\nSecurity note: IAM scoped down",
    "<!-- Ticket: bad -->\nTicket: TRAIG-1\nSecurity note: none (x)",
]
BADS = [
    "",
    None,
    "Ticket: TRAIG-123",
    "Security note: none (x)",
    "Ticket:\nSecurity note: none (x)",
    "Ticket: TRAIG-123\nSecurity note:",
    "Ticket: <KEY-123 | #123 | none (reason)>\nSecurity note: none (x)",
    "Ticket: TRAIG-123\nSecurity note: <one line>",
    "Ticket: none\nSecurity note: none (x)",
    "Ticket: none ()\nSecurity note: none (x)",
    "Ticket: TRAIG-123\nSecurity note: none",
    "Ticket: soon\nSecurity note: none (x)",
    "Ticket: x TRAIG-123\nSecurity note: none (x)",
    "<!--\nTicket: TRAIG-1\nSecurity note: none (x)\n-->",
    "see Ticket: TRAIG-1 and Security note: none (x) inline",
]


class T(unittest.TestCase):
    def test_pass(self):
        for b in OKS:
            self.assertEqual(c.check(b), [], b)

    def test_fail(self):
        for b in BADS:
            self.assertTrue(c.check(b), repr(b))

    def test_crlf_and_bare_cr_bodies_pass(self):
        # GitHub PR bodies can carry CRLF (or bare CR) line endings.
        for b in (
            "Ticket: TRAIG-123\r\nSecurity note: none (docs only)\r\n",
            "Ticket: TRAIG-123\rSecurity note: none (docs only)\r",
        ):
            self.assertEqual(c.check(b), [], repr(b))

    def test_bots_exempt(self):
        for a in ("dependabot[bot]", "renovate[bot]", "github-actions[bot]"):
            self.assertEqual(c.check("", a), [])
        self.assertTrue(c.check("", "dependabot-lookalike"))

    def test_cli_env(self):
        script = os.path.join(os.path.dirname(__file__), "check_pr_ticket_security.py")
        env = dict(
            os.environ,
            PR_BODY="Ticket: TRAIG-1\nSecurity note: none (x)",
            PR_AUTHOR="a",
        )
        self.assertEqual(
            subprocess.run([sys.executable, script], env=env).returncode, 0
        )
        env["PR_BODY"] = "$(touch /tmp/pwn)"
        self.assertEqual(
            subprocess.run([sys.executable, script], env=env).returncode, 1
        )


if __name__ == "__main__":
    unittest.main()
