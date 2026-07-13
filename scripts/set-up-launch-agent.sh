JUST=$(command -v just)
if [ -z "${JUST}" ]; then
    echo "Error: just is not installed or is not on PATH." >&2
    exit 1
fi

OUT_LOG=${HOME}/Library/Logs/bedrock-sso-proxy.out.log
ERR_LOG=${HOME}/Library/Logs/bedrock-sso-proxy.err.log

mkdir -p "${HOME}/Library/LaunchAgents"

cat > "${HOME}/Library/LaunchAgents/local.bedrock-sso-proxy.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>local.bedrock-sso-proxy</string>

    <key>ProgramArguments</key>
    <array>
        <string>${JUST}</string>
        <string>start</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${HOME}/Git/bedrock-sso-proxy</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$(dirname "${JUST}"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>${OUT_LOG}</string>

    <key>StandardErrorPath</key>
    <string>${ERR_LOG}</string>
</dict>
</plist>
EOF

sudo tee /etc/newsyslog.d/bedrock-sso-proxy.conf <<EOF
# logfilename                                          [owner:group]    mode count size when flags
${OUT_LOG}           $(whoami):staff  644  5     1024 *    GN
${ERR_LOG}           $(whoami):staff  644  5     1024 *    GN
EOF