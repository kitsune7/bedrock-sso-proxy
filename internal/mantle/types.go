// Package mantle talks to the OpenAI Responses API hosted on the
// bedrock-mantle endpoint. Models like openai.gpt-5.5 are reachable only
// through this surface — they are not served by Converse, Invoke, or any other
// bedrock-runtime API — so this package covers what the AWS SDK does not.
package mantle

import "encoding/json"

// --- Request ---

// Request is the Responses API request body. It differs from Chat Completions
// in the parts that matter here: `input` instead of `messages`, and
// `max_output_tokens` instead of `max_tokens`.
type Request struct {
	Model           string          `json:"model"`
	Input           []InputItem     `json:"input"`
	Instructions    string          `json:"instructions,omitempty"`
	MaxOutputTokens *int32          `json:"max_output_tokens,omitempty"`
	Temperature     *float32        `json:"temperature,omitempty"`
	TopP            *float32        `json:"top_p,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	Tools           []Tool          `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
}

// InputItem is one element of the `input` array. The Responses API models a
// conversation as a flat list of typed items rather than role-tagged messages:
// a plain message, a function call the model made, or a function call's output.
type InputItem struct {
	Type string `json:"type,omitempty"`

	// Type "message" (or omitted, which the API treats as a message).
	Role    string        `json:"role,omitempty"`
	Content []ContentPart `json:"content,omitempty"`

	// Type "function_call". Arguments has no omitempty: Mantle validates `input`
	// as a strict union and rejects a function_call with the key absent, even
	// for a no-argument function. An empty string is accepted.
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`

	// Type "function_call_output". Same as Arguments — the key must be present
	// even when a tool returned nothing.
	Output string `json:"output"`
}

// ContentPart is a typed content block. Input text uses "input_text"; assistant
// text — both what the model returns and what is replayed back as conversation
// history — uses "output_text".
//
// Text has no omitempty for the same reason as InputItem.Arguments: Mantle
// rejects a content part with no "text" key, so an empty message must serialize
// as "text":"" rather than dropping the field.
type ContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`

	// Set on "input_image" only. Mantle accepts a data: or s3:// URL here — an
	// https URL is rejected — and the field is a bare string, not the nested
	// object Chat Completions uses.
	ImageURL string `json:"image_url,omitempty"`
}

// Tool is a Responses-API tool definition. Unlike Chat Completions, the
// function fields are inlined rather than nested under a "function" key.
type Tool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// --- Response ---

// Response is the non-streaming Responses API result. `output` is a list of
// typed items; assistant text and function calls are siblings there rather
// than fields of a single message.
type Response struct {
	ID                string       `json:"id"`
	Model             string       `json:"model"`
	Status            string       `json:"status"`
	Output            []OutputItem `json:"output"`
	Usage             *Usage       `json:"usage,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

type OutputItem struct {
	Type string `json:"type"`

	// Type "message".
	Role    string        `json:"role,omitempty"`
	Content []ContentPart `json:"content,omitempty"`

	// Type "function_call".
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// --- Streaming ---

// StreamEvent is one SSE event from a streaming response. The Responses API
// names events semantically ("response.output_text.delta") rather than sending
// a single chunk shape, so Type drives the translation.
type StreamEvent struct {
	Type string `json:"type"`

	// Index of the output item this event concerns. Function-call arguments
	// arrive as deltas keyed by this index.
	OutputIndex int `json:"output_index"`

	// Set on "response.output_text.delta" and
	// "response.function_call_arguments.delta".
	Delta string `json:"delta,omitempty"`

	// Set on "response.output_item.added" — carries the function-call name and
	// call_id before its argument deltas start.
	Item *OutputItem `json:"item,omitempty"`

	// Set on "response.completed" and "response.incomplete".
	Response *Response `json:"response,omitempty"`
}
