# Bedrock SSO Proxy

A local OpenAI/Ollama-compatible proxy for Amazon Bedrock models using AWS SSO credentials.

The proxy lets tools that know how to talk to OpenAI's chat completions API, or Ollama's chat API, send requests to Bedrock's Converse and ConverseStream APIs. It uses your configured AWS profile, automatically launches `aws sso login` when credentials are missing or expired, and translates request/response shapes between the client API and Bedrock.

## Features

- OpenAI-compatible `POST /v1/chat/completions`
- OpenAI-compatible `GET /v1/models`
- Ollama-compatible `POST /api/chat`
- Ollama-compatible `GET /api/tags`
- Streaming responses for both OpenAI-style SSE and Ollama-style NDJSON
- Tool/function calling translation
- Basic inference options: `max_tokens`, `temperature`, `top_p`, and `stop`
- Cross-region inference profile ID prefixing
- `.env`, environment variable, and CLI flag configuration

## Requirements

- Go 1.26 or newer
- AWS CLI configured for SSO
- Access to Amazon Bedrock in the selected AWS account and region

Your AWS principal needs permission to invoke the selected Bedrock models, typically including:

```json
{
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Effect": "Allow",
  "Resource": "*"
}
```

## Quick Start

Configure and log in to AWS SSO:

```bash
aws configure sso
aws sso login --profile your-sso-profile
```

Run the proxy:

```bash
go run . -profile your-sso-profile -region us-east-1
```

By default, the server listens on `http://localhost:8000` and uses `claude-opus-4-8` when a request omits `model`.

Check that it is running:

```bash
curl http://localhost:8000/health
```

## Configuration

Configuration can come from CLI flags, environment variables, or a local `.env` file. CLI flags take precedence over environment variables.

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `-port` | `BEDROCK_PROXY_PORT` | `8000` | Local listen port |
| `-profile` | `AWS_PROFILE` | empty | AWS profile name |
| `-region` | `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `-default-model` | `DEFAULT_MODEL` | `claude-opus-4-8` | Model alias or Bedrock model ID used when requests omit `model` |
| `-cross-region` | `CROSS_REGION` | `true` | Prefix model IDs with the region family, such as `us.` |
| `-verbose` | `VERBOSE` | `false` | Log model resolution details |

Example `.env`:

```bash
AWS_PROFILE=your-sso-profile
AWS_REGION=us-east-1
DEFAULT_MODEL=claude-sonnet-5
BEDROCK_PROXY_PORT=8000
```

## OpenAI-Compatible Usage

List models:

```bash
curl http://localhost:8000/v1/models
```

Create a chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-5",
    "messages": [
      { "role": "user", "content": "Write a haiku about local proxies." }
    ],
    "max_tokens": 200
  }'
```

Stream a chat completion:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-5",
    "stream": true,
    "stream_options": { "include_usage": true },
    "messages": [
      { "role": "user", "content": "Explain Bedrock Converse in one paragraph." }
    ]
  }'
```

Point an OpenAI-compatible client at the proxy by setting its base URL to:

```text
http://localhost:8000/v1
```

The proxy does not validate an OpenAI API key. If your client requires one, use any placeholder value.

## Ollama-Compatible Usage

List models:

```bash
curl http://localhost:8000/api/tags
```

Create a chat completion:

```bash
curl http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-5:latest",
    "stream": false,
    "messages": [
      { "role": "user", "content": "Write a haiku about local proxies." }
    ],
    "options": {
      "num_predict": 200
    }
  }'
```

For Ollama clients, set the host/base URL to:

```text
http://localhost:8000
```

## Models

Model aliases are defined in `internal/models/registry.go`. The model registry maps friendly names such as `claude-sonnet-5` to Bedrock model or inference profile IDs.

There are known gaps in the default registry. Add models as needed by editing `defaultModels` in `internal/models/registry.go`:

```go
{Alias: "my-model-alias", BedrockID: "anthropic.example-model-id", OwnedBy: "anthropic", CrossRegion: true},
```

To find the inference profile ID for a Bedrock model available to your account, run:

```bash
aws bedrock list-inference-profiles --region us-east-1 --profile your-sso-profile
```

Use the returned inference profile ID as the `BedrockID`. If `CrossRegion` is `true` and `-cross-region` is enabled, the proxy prefixes IDs without an existing region family prefix using the first segment of `AWS_REGION`:

- `us-east-1` becomes `us.`
- `eu-west-1` becomes `eu.`
- `ap-southeast-1` becomes `ap.`

Requests can also pass a full Bedrock ID directly. Unknown model names are passed through, with the optional region prefix applied when cross-region mode is enabled.

## Supported Request Shape

The OpenAI-compatible endpoint supports:

- Roles: `system`, `developer`, `user`, `assistant`, and `tool`
- Text message content
- Text content parts
- Tool/function definitions and tool choice
- Assistant tool calls and tool results
- `max_tokens`, `temperature`, `top_p`, and `stop`
- Streaming with optional usage chunks

The Ollama-compatible endpoint supports:

- Chat messages
- Streaming by default, matching Ollama behavior
- `options.num_predict`, `options.temperature`, `options.top_p`, and `options.stop`
- Tools and tool calls

Image inputs and other multimodal content are currently ignored by the translation layer.

## Development

Run tests:

```bash
go test ./...
```

Useful files:

- `main.go` wires routes and middleware.
- `internal/config/config.go` parses flags, environment variables, and `.env`.
- `internal/auth/sso.go` loads AWS credentials and renews expired SSO sessions.
- `internal/models/registry.go` defines model aliases and Bedrock IDs.
- `internal/bedrock/translate.go` translates OpenAI-shaped requests/responses to and from Bedrock Converse.
- `internal/handler/chat.go` implements the OpenAI-compatible endpoints.
- `internal/handler/ollama.go` implements the Ollama-compatible endpoints.
