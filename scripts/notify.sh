#!/bin/bash
# Notification script for PipNN-DiskANN cron job
# Supports macOS notification center and terminal output

TITLE="$1"
MESSAGE="$2"
LEVEL="${3:-INFO}"  # INFO, WARNING, ERROR

echo "================================"
echo "[$LEVEL] $TITLE"
echo "================================"
echo "$MESSAGE"
echo "================================"

# macOS Notification Center
if command -v osascript &>/dev/null; then
    osascript -e 'display notification "'"$TITLE'" with message "'"$MESSAGE'" sound name "Glass"'
elif command -v notify-send &>/dev/null; then
    notify-send "$TITLE" "$MESSAGE"
fi

# Optional: Send to a log file
NOTIFICATION_LOG="/Users/ryan/Code/knowhere/logs/pipnn-cron/notifications.log"
mkdir -p "$(dirname "$NOTIFICATION_LOG")"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$LEVEL] $TITLE: $MESSAGE" >> "$NOTIFICATION_LOG"