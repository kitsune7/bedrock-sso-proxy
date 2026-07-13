start:
  go run ./main.go

# Mac-specific
set-up-launch-agent:
  ./scripts/set-up-launch-agent.sh

start-launch-agent:
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.bedrock-sso-proxy.plist
  launchctl kickstart -k gui/$(id -u)/local.bedrock-sso-proxy

stop-launch-agent:
  launchctl bootout gui/$(id -u)/local.bedrock-sso-proxy

check-launch-agent:
  launchctl print   gui/$(id -u)/local.bedrock-sso-proxy
