# Bedrock SSO Proxy

A local OpenAI/Ollama-compatible proxy for Amazon Bedrock models using AWS SSO credentials.

The proxy lets tools that know how to talk to OpenAI's chat completions API, or Ollama's chat API, send requests to Bedrock. It uses your configured AWS profile, automatically launches `aws sso login` when credentials are missing or expired, and translates request/response shapes between the client API and Bedrock.

Not every Bedrock model speaks the same API. Claude, GPT-6 Astra, and gpt-oss are served by Converse/ConverseStream on `bedrock-runtime`; the GPT-5 family is served only by the OpenAI Responses API on the `bedrock-mantle` endpoint. The client does not need to care: the model name alone selects the backend, and the translation layer adapts parameters to whatever that model accepts.

## Features

- OpenAI-compatible `POST /v1/chat/completions`
- OpenAI-compatible `GET /v1/models`
- Ollama-compatible `POST /api/chat`
- Ollama-compatible `GET /api/tags`
- Streaming responses for both OpenAI-style SSE and Ollama-style NDJSON
- Tool/function calling translation, including multi-turn tool results
- Basic inference options: `max_tokens`, `temperature`, `top_p`, and `stop`
- Backend routing by model: Bedrock Converse or the Bedrock Mantle Responses API
- Per-model parameter adaptation, so a request that a model would reject is fixed rather than failed
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

The GPT-5 models go through a different service, so they need their own action:

```json
{
  "Action": "bedrock-mantle:CreateInference",
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

By default, the server listens on `http://localhost:8000` and uses `gpt-5.6-sol` when a request omits `model`.

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
| `-default-model` | `DEFAULT_MODEL` | `gpt-5.6-sol` | Model alias or Bedrock model ID used when requests omit `model` |
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

Model aliases are defined in `internal/models/registry.go`. The registry maps friendly names such as `claude-sonnet-5` to Bedrock model or inference profile IDs, and also declares which API serves the model and which parameters it accepts. `GET /v1/models` lists everything registered.

Anthropic models, `gpt-6-astra`, and `gpt-oss-120b` / `gpt-oss-20b` are served by Converse. Astra resolves to the geographic cross-region inference profile (for example, `us.openai.gpt-6-astra`) and supports `bedrock-runtime` from all documented regions. The GPT-5 family (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`) is served only by the Responses API on Bedrock Mantle — there is no Converse or Invoke path for them, they have no cross-region inference profiles, and they are in-region only: `us-east-1` and `us-east-2` for all of them, plus `us-west-2` for `gpt-5.6-terra` and `gpt-5.6-luna`. Set `-region` accordingly if you want to use them.

There are known gaps in the default registry. Add models as needed by editing `defaultModels` in `internal/models/registry.go`:

```go
{Alias: "my-model-alias", BedrockID: "anthropic.example-model-id", OwnedBy: "anthropic", CrossRegion: true},
```

Each entry can declare traits that the translation layer acts on:

| Field | Effect |
| --- | --- |
| `Backend` | Which API serves the model. Empty means `converse`; `mantle-responses` routes to the Responses API on Bedrock Mantle. |
| `CrossRegion` | Whether the ID may take a region family prefix such as `us.`. |
| `NoSampling` | The model rejects `temperature` and `top_p`, so both are dropped. |
| `ExclusiveSampling` | The model accepts either `temperature` or `top_p` but returns a 400 if both are sent. `temperature` wins and `top_p` is dropped. |
| `Reasoning` | The model spends part of its output budget on internal reasoning that never reaches the client, so a very small `max_tokens` would return empty content. `max_tokens` is raised to at least `ReasoningMinOutputTokens` (4096). |

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

On the Mantle Responses backend, `system` and `developer` messages become the Responses `instructions` field, and `stop` is dropped because the Responses API has no equivalent.

## Development

Run tests:

```bash
go test ./...
```

Useful files:

- `main.go` wires routes and middleware.
- `internal/config/config.go` parses flags, environment variables, and `.env`.
- `internal/auth/sso.go` loads AWS credentials and renews expired SSO sessions.
- `internal/models/registry.go` defines model aliases, Bedrock IDs, backend routing, and per-model parameter traits.
- `internal/bedrock/translate.go` translates OpenAI-shaped requests/responses to and from Bedrock Converse.
- `internal/mantle/` talks to the OpenAI Responses API on the `bedrock-mantle` endpoint: `client.go` signs the request, `translate.go` converts request/response shapes, `stream.go` decodes the SSE event stream.
- `internal/handler/chat.go` implements the OpenAI-compatible endpoints.
- `internal/handler/ollama.go` implements the Ollama-compatible endpoints.
- `internal/handler/chat_mantle.go` and `internal/handler/ollama_mantle.go` serve the Mantle backend for each surface.

## License

MIT
