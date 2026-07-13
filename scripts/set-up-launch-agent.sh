#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${PROJECT_DIR}/.env"
    set +a
fi

JUST=$(command -v just || true)
if [ -z "${JUST}" ]; then
    echo "Error: just is not installed or is not on PATH." >&2
    exit 1
fi

if [ -z "${AWS_PROFILE:-}" ]; then
    echo "Error: AWS_PROFILE is not set." >&2
    echo "Set AWS_PROFILE in ${PROJECT_DIR}/.env or export it before running this script." >&2
    exit 1
fi

AWS_REGION=${AWS_REGION:-us-east-1}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-${AWS_REGION}}
AWS_CONFIG_FILE=${AWS_CONFIG_FILE:-${HOME}/.aws/config}
AWS_SHARED_CREDENTIALS_FILE=${AWS_SHARED_CREDENTIALS_FILE:-${HOME}/.aws/credentials}

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
    <string>${PROJECT_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>PATH</key>
        <string>$(dirname "${JUST}"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>AWS_PROFILE</key>
        <string>${AWS_PROFILE}</string>
        <key>AWS_REGION</key>
        <string>${AWS_REGION}</string>
        <key>AWS_DEFAULT_REGION</key>
        <string>${AWS_DEFAULT_REGION}</string>
        <key>AWS_CONFIG_FILE</key>
        <string>${AWS_CONFIG_FILE}</string>
        <key>AWS_SHARED_CREDENTIALS_FILE</key>
        <string>${AWS_SHARED_CREDENTIALS_FILE}</string>
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