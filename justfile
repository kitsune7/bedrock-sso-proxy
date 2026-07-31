set dotenv-load

start:
  go run ./main.go

heartbeat model="":
  @port="${BEDROCK_PROXY_PORT:-8000}"; \
    model="{{model}}"; \
    curl -fsS -X POST "http://localhost:${port}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"'"${model}"'","stream":false,"messages":[{"role":"user","content":"Reply with exactly: ok"}]}'

# Mac-specific
set-up-launch-agent:
  ./scripts/set-up-launch-agent.sh

start-launch-agent:
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.bedrock-sso-proxy.plist
  launchctl kickstart -k gui/$(id -u)/local.bedrock-sso-proxy

stop-launch-agent:
  launchctl bootout gui/$(id -u)/local.bedrock-sso-proxy

# Reload the agent to pick up code changes (it runs `just start`, i.e. `go run`)
restart-launch-agent:
  # `-` so a restart still works when the agent is not currently loaded.
  -launchctl bootout gui/$(id -u)/local.bedrock-sso-proxy
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.bedrock-sso-proxy.plist
  launchctl kickstart -k gui/$(id -u)/local.bedrock-sso-proxy

check-launch-agent:
  launchctl print   gui/$(id -u)/local.bedrock-sso-proxy

check-logs:
  echo "Checking output logs..."; \
  tail ~/Library/Logs/bedrock-sso-proxy.out.log; \
  echo "Checking error logs..."; \
  tail ~/Library/Logs/bedrock-sso-proxy.err.log