# Security Guide for Traigent

## Overview

This document outlines the security improvements implemented in Traigent and provides guidance for secure usage of the framework.

## Recent Security Improvements

### 1. Elimination of exec() Usage

**Previous Risk**: The original `SeamlessParameterProvider` used `exec()` for dynamic code injection, creating a critical security vulnerability that could allow arbitrary code execution.

**Solution Implemented**: Complete replacement with AST-based transformation using `SafeSeamlessProvider`.

#### Key Benefits:
- **No arbitrary code execution**: AST transformation is inherently safer than exec()
- **Input validation**: All configuration values are validated before injection
- **Type safety**: Only safe, simple types can be injected (strings, numbers, lists, dicts)
- **Audit trail**: All transformations are logged for security monitoring

### 2. Complete Authentication Implementations

All authentication methods that were marked as TODO have been fully implemented:

#### SAML Authentication
- Full SAML 2.0 support using python3-saml
- SP and IdP configuration validation
- Secure session management
- Logout support

#### OpenID Connect (OIDC)
- JWT token validation with PyJWT
- JWKS endpoint support for key rotation
- Claims validation and user mapping
- Access token verification

#### TOTP (Time-based One-Time Password)
- Authenticator app support (Google Authenticator, Authy, etc.)
- QR code provisioning URI generation
- Backup codes for account recovery
- Configurable time window for clock skew

#### SMS Authentication
- Twilio integration for SMS delivery
- Rate limiting and attempt tracking
- Expiring verification codes (5 minutes)
- Phone number validation

### 3. Centralized Credential Management

A new secure credential store has been implemented with:
- **Encryption at rest**: All credentials encrypted using Fernet (symmetric encryption)
- **Master key management**: Secure generation and storage of encryption keys
- **Environment variable support**: Seamless integration with existing deployment practices
- **Credential rotation**: Built-in support for key rotation with history tracking
- **Audit logging**: All credential operations are logged

## Migration Guide

### Migrating from SeamlessParameterProvider to SafeSeamlessProvider

The new `SafeSeamlessProvider` is a drop-in replacement for the original provider. To migrate:

#### Option 1: Automatic Migration (Recommended)
```python
from traigent.config.providers_safe import migrate_to_safe_provider

# This replaces the provider globally
migrate_to_safe_provider("seamless")
```

#### Option 2: Manual Migration
```python
from traigent.config.providers_safe import SafeSeamlessProvider

# Use in your optimization decorators
@traigent.optimize(
    injection_mode="seamless",  # Same as before
    configuration_space={"model": ["gpt-3.5", "gpt-4"]}
)
def my_function():
    model = "gpt-3.5"  # Will be safely overridden
    return f"Using {model}"
```

### Compatibility Notes

The `SafeSeamlessProvider` maintains 100% API compatibility with the original provider. Your existing code will continue to work without modifications. The only difference is the elimination of security vulnerabilities.

## Authentication Setup

### SAML Configuration

```python
from traigent.security.auth import SAMLAuthProvider

saml_settings = {
    "sp": {
        "entityId": "https://yourapp.com",
        "assertionConsumerService": {
            "url": "https://yourapp.com/saml/acs",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        }
    },
    "idp": {
        "entityId": "https://idp.example.com",
        "singleSignOnService": {
            "url": "https://idp.example.com/sso",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "x509cert": "YOUR_IDP_CERTIFICATE"
    }
}

provider = SAMLAuthProvider(saml_settings)
```

### OIDC Configuration

```python
from traigent.security.auth import OIDCAuthProvider

oidc_settings = {
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "issuer": "https://accounts.google.com",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token"
}

provider = OIDCAuthProvider(oidc_settings)
```

### TOTP Setup

```python
from traigent.security.auth import TOTPAuthProvider

totp_provider = TOTPAuthProvider(issuer_name="YourApp")

# Generate secret for user
secret = totp_provider.generate_totp_secret()

# Generate QR code URI
qr_uri = totp_provider.generate_provisioning_uri("user@example.com", secret)

# Verify code from user
is_valid = totp_provider.verify_totp_code("123456", secret)
```

### SMS Authentication

```python
from traigent.security.auth import SMSAuthProvider

sms_settings = {
    "account_sid": "YOUR_TWILIO_ACCOUNT_SID",
    "auth_token": "YOUR_TWILIO_AUTH_TOKEN",
    "from_number": "+1234567890"
}

provider = SMSAuthProvider(sms_settings)

# Send verification code
message_sid = provider.send_verification_code("+1987654321", "user_id_123")

# Verify code
is_valid = provider.verify_sms_code("user_id_123", "123456")
```

## Credential Management Best Practices

### 1. Initial Setup

```python
from traigent.security.credentials import get_credential_store, get_api_key_manager
from traigent.security.credentials import CredentialType

# Initialize credential store with custom path and master key
store = get_credential_store()

# Store API keys securely
store.set("OPENAI_API_KEY", "sk-...", CredentialType.API_KEY)
store.set("DATABASE_PASSWORD", "secure_password", CredentialType.PASSWORD)
```

### 2. Environment Variables

Set environment variables for automatic loading:
```bash
export TRAIGENT_MASTER_KEY="your-master-encryption-key"
export TRAIGENT_OPENAI_API_KEY="sk-..."
export TRAIGENT_DATABASE_PASSWORD="secure_password"
```

### 3. Using the API Key Manager

```python
from traigent.security.credentials import get_api_key_manager

manager = get_api_key_manager()

# Get API keys (checks env vars first, then encrypted store)
openai_key = manager.get_openai_key()
anthropic_key = manager.get_anthropic_key()

# Validate all keys
validation = manager.validate_api_keys()
print(validation)  # {'openai': True, 'anthropic': False, ...}
```

### 4. Credential Rotation

```python
# Rotate a credential (keeps history)
store.rotate_credential("DATABASE_PASSWORD", "new_secure_password")

# Export template for environment variables
template = store.export_env_template()
with open(".env.template", "w") as f:
    f.write(template)
```

## Security Checklist

### For Developers

- [ ] Migrate all uses of `SeamlessParameterProvider` to `SafeSeamlessProvider`
- [ ] Never hardcode credentials in source code
- [ ] Use the credential store for all sensitive data
- [ ] Enable MFA for user accounts (TOTP or SMS)
- [ ] Implement proper session management with SAML/OIDC
- [ ] Regularly rotate API keys and credentials
- [ ] Review and validate all configuration inputs
- [ ] Enable audit logging for security events

### For Deployment

- [ ] Store master encryption key securely (e.g., AWS KMS, Azure Key Vault)
- [ ] Use environment variables for production credentials
- [ ] Enable TLS/SSL for all network communications
- [ ] Implement rate limiting for authentication endpoints
- [ ] Set up monitoring and alerting for security events
- [ ] Regular security audits and dependency updates
- [ ] Implement proper backup and recovery procedures

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting Security Issues

If you discover a security vulnerability in Traigent, please report it to:
- Email: security@traigent.ai
- Do not create public GitHub issues for security vulnerabilities
- Include detailed steps to reproduce the issue
- Include the potential impact and any suggested fixes
- We aim to respond within 48 hours and provide updates every 7 days until resolution

## Dependencies

The security features require the following optional dependencies:

```bash
# For SAML authentication
pip install python3-saml

# For OIDC authentication
pip install pyjwt[crypto]

# For TOTP authentication
pip install pyotp

# For SMS authentication
pip install twilio

# For encryption (usually included)
pip install cryptography
```

## Security Testing

Run the security test suite to verify all security features:

```bash
# Run security tests
pytest tests/security/ -v

# Run specific security test categories
pytest tests/security/test_seamless_security.py -v  # AST transformation tests
pytest tests/security/test_auth_complete.py -v      # Authentication tests
pytest tests/security/test_credentials.py -v        # Credential management tests
```

All security tests should pass before deployment.

## Conclusion

These security improvements significantly enhance Traigent's security posture by:
1. Eliminating critical code injection vulnerabilities
2. Providing enterprise-grade authentication options
3. Ensuring secure credential management
4. Maintaining backward compatibility

For questions or additional security guidance, please refer to the Traigent documentation or contact the security team.
