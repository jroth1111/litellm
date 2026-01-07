# Experimental OAuth Flows

> [!WARNING]
> Some OAuth integration methods in this fork are experimental and may conflict with provider Terms of Service.

## Supported OAuth Flows (Recommended)

These flows use official OAuth endpoints and are fully supported:

| Provider | Flow | Status |
|----------|------|--------|
| **Anthropic** | Browser OAuth | ✅ Supported |
| **OpenAI** | Browser OAuth | ✅ Supported |
| **Google AI** | Browser OAuth | ✅ Supported |
| **GitHub Copilot** | Device Code | ✅ Supported |
| **Qwen** | Device Code | ✅ Supported |
| **Cursor** | Browser OAuth | ✅ Supported |

## Experimental Flows (Use at Your Own Risk)

> [!CAUTION]
> Cookie-based OAuth flows extract session tokens from browser cookies. These methods:
> - May violate provider Terms of Service
> - Could result in account suspension
> - Are intended **only for personal experimentation**
> - Should **never** be used in production or shared environments

### Files with Experimental Code

- `litellm/llms/anthropic/cookie_oauth.py` - Cookie extraction for Claude
- Any adapter referencing session/cookie-based auth

### Legal Considerations

1. **Review ToS**: Before using any OAuth flow, review the provider's Terms of Service
2. **Personal Use Only**: Experimental flows are for personal learning and development
3. **No Warranties**: These flows may break at any time without notice
4. **Rate Limits**: Using subscription tokens may be subject to different rate limits than API access

## Configuration

To disable experimental flows entirely, do not configure cookie-based adapters in your auth settings. Use only the official `litellm auth login <provider>` CLI flows.
