#!/bin/bash
# Launch Chrome with remote debugging enabled
# This allows Playwright to connect to your existing browser session

echo "=============================================================================="
echo "Launch Chrome with Remote Debugging"
echo "=============================================================================="
echo ""
echo "This will launch Chrome with remote debugging on port 9222."
echo "Your existing Chrome will be closed and restarted."
echo ""

# Check if Chrome is already running
if pgrep -x "Chrome" > /dev/null; then
    echo "⚠️  Chrome is currently running."
    echo ""
    read -p "Do you want to close it and restart with debugging? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Closing existing Chrome..."
        pkill -x "Chrome"
        sleep 1
    else
        echo "Exiting. Please close Chrome manually and run this script again."
        exit 1
    fi
fi

echo "Starting Chrome with remote debugging..."
echo ""

# Launch Chrome with remote debugging
# Using a temporary user data dir to avoid conflicts
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-debug-profile \
  https://twitter.com &

sleep 2

# Check if Chrome started
if pgrep -x "Chrome" > /dev/null; then
    echo "✓ Chrome started successfully!"
    echo ""
    echo "Chrome is running with remote debugging on port 9222."
    echo "You can now run the test script:"
    echo ""
    echo "  python3 scripts/twitter/test_existing_browser.py"
    echo ""
else
    echo "❌ Failed to start Chrome"
    exit 1
fi
