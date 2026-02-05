"""
Key management for Twitter Growth Agent.

Parses API keys from ~/.devease/keys file with helpful setup instructions.
"""

import os
from pathlib import Path
from typing import Optional

from .types import TwitterKeys


class KeyManager:
    """
    Parse and manage API keys from ~/.devease/keys file.

    Provides helpful instructions if file is missing.
    """

    DEFAULT_KEYS_PATH = Path.home() / ".devease" / "keys"

    def __init__(self, keys_path: Optional[Path] = None):
        """
        Initialize KeyManager.

        Args:
            keys_path: Path to keys file (defaults to ~/.devease/keys)
        """
        self.keys_path = keys_path or self.DEFAULT_KEYS_PATH
        self._keys: Optional[TwitterKeys] = None

    def load_keys(self) -> TwitterKeys:
        """
        Load and parse keys from file.

        Returns:
            TwitterKeys object with parsed credentials

        Raises:
            FileNotFoundError: If keys file doesn't exist
        """
        if not self.keys_path.exists():
            self._print_setup_instructions()
            raise FileNotFoundError(
                f"\nKeys file not found at {self.keys_path}\n"
                f"Please create it with the instructions above."
            )

        env_vars = {}
        with open(self.keys_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        # Expand ~ in paths
        cookies_path = env_vars.get('TWITTER_COOKIES_PATH', '~/.devease/twitter_cookies.json')
        data_path = env_vars.get('DEVEASE_DATA_PATH', '~/devease/data')

        self._keys = TwitterKeys(
            openai_api_key=env_vars.get('OPENAI_API_KEY', ''),
            openai_org_id=env_vars.get('OPENAI_ORG_ID'),
            twitter_api_key=env_vars.get('TWITTER_API_KEY'),
            twitter_api_secret=env_vars.get('TWITTER_API_SECRET'),
            twitter_access_token=env_vars.get('TWITTER_ACCESS_TOKEN'),
            twitter_access_secret=env_vars.get('TWITTER_ACCESS_SECRET'),
            twitter_bearer_token=env_vars.get('TWITTER_BEARER_TOKEN'),
            twitter_cookies_path=os.path.expanduser(cookies_path),
            devease_data_path=os.path.expanduser(data_path)
        )

        return self._keys

    def _print_setup_instructions(self):
        """Print helpful instructions for creating the keys file."""
        print("\n" + "="*70)
        print("Twitter Growth Agent - Keys Setup")
        print("="*70)
        print(f"\nKeys file not found: {self.keys_path}")
        print("\nStep 1: Create the directory")
        print(f"  mkdir -p {self.keys_path.parent}")
        print("\nStep 2: Create the keys file")
        print(f"  cat > {self.keys_path} << 'EOF'")
        print("# Twitter Growth Agent Keys")
        print("# Format: KEY_NAME=value")
        print("")
        print("# REQUIRED: OpenAI API (for content generation)")
        print("OPENAI_API_KEY=sk-your-openai-api-key-here")
        print("OPENAI_ORG_ID=org-your-org-id-here  # Optional")
        print("")
        print("# OPTIONAL: Twitter API (for analytics/metrics)")
        print("TWITTER_API_KEY=your-twitter-api-key-here")
        print("TWITTER_API_SECRET=your-twitter-api-secret-here")
        print("TWITTER_ACCESS_TOKEN=your-twitter-access-token-here")
        print("TWITTER_ACCESS_SECRET=your-twitter-access-secret-here")
        print("TWITTER_BEARER_TOKEN=your-twitter-bearer-token-here")
        print("")
        print("# OPTIONAL: Browser session cookies")
        print("TWITTER_COOKIES_PATH=~/.devease/twitter_cookies.json")
        print("")
        print("# OPTIONAL: DevEase data paths")
        print("DEVEASE_DATA_PATH=~/devease/data")
        print("EOF")
        print("\nStep 3: Set restrictive permissions")
        print(f"  chmod 600 {self.keys_path}")
        print("\nStep 4: Edit the file and add your actual API keys")
        print(f"  nano {self.keys_path}")
        print("\n" + "="*70)

    @property
    def keys(self) -> TwitterKeys:
        """
        Get loaded keys (lazy load).

        Returns:
            TwitterKeys object
        """
        if self._keys is None:
            self.load_keys()
        return self._keys

    def validate_required_keys(self) -> bool:
        """
        Check that required keys are present.

        Returns:
            True if all required keys are present

        Raises:
            ValueError: If required keys are missing or invalid
        """
        keys = self.keys
        if not keys.openai_api_key or keys.openai_api_key.startswith('sk-') == False or len(keys.openai_api_key) < 20:
            raise ValueError(
                f"OPENAI_API_KEY is required but not properly set in {self.keys_path}\n"
                f"Current value: {keys.openai_api_key[:20] if keys.openai_api_key else 'empty'}..."
            )
        return True
