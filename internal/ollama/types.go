package ollama

import "encoding/json"

// ChatRequest is the Ollama /api/chat request body.
// Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
type ChatRequest struct {
	Model    string          `json:"model"`
	Messages []Message       `json:"messages"`
	Stream   *bool           `json:"stream,omitempty"` // defaults to true when omitted
	Format   json.RawMessage `json:"format,omitempty"`
	Options  *Options        `json:"options,omitempty"`
	Tools    []Tool          `json:"tools,omitempty"`
	// KeepAlive, Template, etc. are accepted but ignored.
}

// StreamEnabled returns true when the client wants streaming (the Ollama default).
func (r *ChatRequest) StreamEnabled() bool {
	if r.Stream == nil {
		return true
	}
	return *r.Stream
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type Tool struct {
	Type     string      `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Options mirrors the subset of Ollama sampling options that map to Bedrock inference config.
type Options struct {
	NumPredict  *int32   `json:"num_predict,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// ChatResponse is one frame of the Ollama chat response.
// Streaming responses emit a sequence of these as line-delimited JSON, with `done: true` on the final frame.
// Non-streaming responses emit exactly one frame with `done: true`.
type ChatResponse struct {
	Model     string  `json:"model"`
	CreatedAt string  `json:"created_at"`
	Message   Message `json:"message"`
	Done      bool    `json:"done"`
	DoneReason string `json:"done_reason,omitempty"`

	// Usage metrics — populated on the final frame. Durations are nanoseconds.
	TotalDuration   int64 `json:"total_duration,omitempty"`
	LoadDuration    int64 `json:"load_duration,omitempty"`
	PromptEvalCount int   `json:"prompt_eval_count,omitempty"`
	EvalCount       int   `json:"eval_count,omitempty"`
}

// TagsResponse is the body of Ollama's /api/tags endpoint.
type TagsResponse struct {
	Models []TagModel `json:"models"`
}

type TagModel struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Size       int64  `json:"size"`
	Digest     string `json:"digest"`
}

// ErrorResponse is the Ollama-style error body.
type ErrorResponse struct {
	Error string `json:"error"`
}
