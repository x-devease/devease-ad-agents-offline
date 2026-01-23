#!/bin/bash

# Install Jupyter Notebook
echo "Installing Jupyter Notebook..."
pip3 install notebook

# Get Python version to determine bin path
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
PYTHON_BIN_PATH="$HOME/Library/Python/${PYTHON_VERSION}/bin"

# Add Python bin directory to PATH in .zshrc if not already present
if [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "Library/Python/${PYTHON_VERSION}/bin" "$HOME/.zshrc"; then
        echo "Adding Python bin directory to PATH in ~/.zshrc..."
        echo "export PATH=\"\$HOME/Library/Python/${PYTHON_VERSION}/bin:\$PATH\"" >> "$HOME/.zshrc"
        echo "✓ Added to ~/.zshrc"
    else
        echo "✓ PATH already configured in ~/.zshrc"
    fi
else
    echo "Creating ~/.zshrc and adding PATH..."
    echo "export PATH=\"\$HOME/Library/Python/${PYTHON_VERSION}/bin:\$PATH\"" > "$HOME/.zshrc"
    echo "✓ Created ~/.zshrc with PATH configuration"
fi

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -m notebook --version > /dev/null 2>&1; then
    echo "✓ Jupyter Notebook installed successfully"
    python3 -m notebook --version 2>&1 | grep -v "NotOpenSSLWarning" | head -1
else
    echo "✗ Installation verification failed"
    exit 1
fi

echo ""
echo "Installation complete!"
echo "To use 'jupyter' command, either:"
echo "  1. Restart your terminal, or"
echo "  2. Run: source ~/.zshrc"
echo ""
echo "You can also use: python3 -m notebook"
