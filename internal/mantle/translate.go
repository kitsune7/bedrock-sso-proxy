package mantle

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"

	"bedrock-sso-proxy/internal/imageref"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"
)

// TranslateRequest converts the internal OpenAI Chat Completions request into a
// Responses API request.
//
// The two APIs differ structurally, not just in field names. Chat Completions
// has role-tagged messages with tool calls attached to the assistant message;
// Responses has a flat list of typed items where a function call and its output
// are siblings of the messages around them. System messages become the
// top-level `instructions` field.
//
// ctx bounds the image fetches an image_url content part may require.
func TranslateRequest(ctx context.Context, req *openai.ChatCompletionRequest, m models.Resolved) (*Request, error) {
	out := &Request{
		Model:  m.ID,
		Stream: req.Stream,
		// Responses spells the output budget differently, and reasoning models
		// need the floor applied for the same reason they do on Converse.
		MaxOutputTokens: m.AdjustMaxTokens(req.MaxTokens),
	}

	if !m.NoSampling {
		out.Temperature = req.Temperature
		out.TopP = req.TopP
	}

	// `stop` has no equivalent on Responses. Dropping it silently is wrong — a
	// client relying on a stop sequence would get over-long output with no
	// indication why — but failing the request is worse for a parameter this
	// incidental. The proxy is lenient and the caller sees the full completion.

	var instructions string
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			if text := msg.ContentString(); text != "" {
				if instructions != "" {
					instructions += "\n\n"
				}
				instructions += text
			}

		case "user":
			content, err := userContent(ctx, msg)
			if err != nil {
				return nil, err
			}
			out.Input = append(out.Input, InputItem{
				Type:    "message",
				Role:    "user",
				Content: content,
			})

		case "assistant":
			if text := msg.ContentString(); text != "" {
				out.Input = append(out.Input, InputItem{
					Type: "message",
					Role: "assistant",
					// Assistant history uses "output_text", not "input_text" —
					// Mantle's union rejects an assistant message carrying an
					// input_text part, which breaks every multi-turn request.
					Content: []ContentPart{{Type: "output_text", Text: text}},
				})
			}
			for _, tc := range msg.ToolCalls {
				out.Input = append(out.Input, InputItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				})
			}

		case "tool":
			out.Input = append(out.Input, InputItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: msg.ContentString(),
			})

		default:
			return nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}
	out.Instructions = instructions

	// Responses requires at least one input item — a request carrying only
	// system/developer messages yields none. Report it as the client error it is
	// rather than letting Mantle's 400 surface as a 502.
	if len(out.Input) == 0 {
		return nil, fmt.Errorf("request has no user, assistant, or tool messages")
	}

	for _, t := range req.Tools {
		if t.Type != "function" {
			continue
		}
		// Responses inlines the function fields rather than nesting them under
		// a "function" key.
		out.Tools = append(out.Tools, Tool{
			Type:        "function",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	if toolChoice, err := translateToolChoice(req.ToolChoice); err != nil {
		return nil, err
	} else if toolChoice != nil {
		out.ToolChoice = toolChoice
	}

	return out, nil
}

// userContent builds the content parts for a user message, resolving any
// image_url parts into the data: URL Mantle requires. Chat Completions nests the
// URL under an object and allows https; Mantle takes a bare string and allows
// only data: and s3://, so an https image has to be fetched and inlined.
func userContent(ctx context.Context, msg openai.Message) ([]ContentPart, error) {
	parts := msg.Parts()
	// A message with no parseable content still has to carry one part — Mantle
	// rejects a message with an empty content array.
	if len(parts) == 0 {
		return []ContentPart{{Type: "input_text"}}, nil
	}

	var out []ContentPart
	for _, p := range parts {
		switch p.Type {
		case "text":
			out = append(out, ContentPart{Type: "input_text", Text: p.Text})

		case "image_url":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return nil, fmt.Errorf("image_url content part has no url")
			}
			ref, err := imageref.Resolve(ctx, nil, p.ImageURL.URL)
			if err != nil {
				return nil, fmt.Errorf("image_url content part: %w", err)
			}
			url := ref.S3URI
			if url == "" {
				url = ref.DataURL()
			}
			out = append(out, ContentPart{Type: "input_image", ImageURL: url})

		default:
			// Audio and file parts have no Responses equivalent here. Failing is
			// better than silently dropping content the caller believes was sent.
			return nil, fmt.Errorf("unsupported content part type: %s", p.Type)
		}
	}
	return out, nil
}

// translateToolChoice maps Chat Completions tool_choice to the Responses form.
// The string values ("auto", "required", "none") are identical; the object form
// drops the nested "function" wrapper.
func translateToolChoice(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return raw, nil
	}
	var obj struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil || obj.Function.Name == "" {
		return nil, fmt.Errorf("unsupported tool_choice: %s", raw)
	}
	return json.Marshal(map[string]string{"type": "function", "name": obj.Function.Name})
}

// TranslateResponse converts a Responses result back into the internal Chat
// Completions shape. model is echoed as the client named it, matching what the
// Converse path does.
func TranslateResponse(resp *Response, model string) *openai.ChatCompletionResponse {
	text, toolCalls := collectOutput(resp.Output)

	msg := openai.Message{Role: "assistant"}
	if text != "" {
		contentJSON, _ := json.Marshal(text)
		msg.Content = contentJSON
	}
	msg.ToolCalls = toolCalls

	out := &openai.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.Choice{{
			Index:        0,
			Message:      msg,
			FinishReason: FinishReason(resp, len(toolCalls) > 0),
		}},
	}

	if resp.Usage != nil {
		out.Usage = openai.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return out
}

// collectOutput flattens the typed output items into the text and tool calls a
// Chat Completions message holds. "reasoning" items are skipped: they have no
// Chat Completions equivalent, and their tokens are already counted in usage.
func collectOutput(items []OutputItem) (string, []openai.ToolCall) {
	var text string
	var toolCalls []openai.ToolCall

	for _, item := range items {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					text += part.Text
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		}
	}

	return text, toolCalls
}

// FinishReason maps a Responses status to a Chat Completions finish_reason.
// Tool calls take precedence: Responses reports status "completed" when the
// model stops to call a function, but Chat Completions clients branch on
// "tool_calls" to decide whether to execute one.
func FinishReason(resp *Response, hasToolCalls bool) *string {
	reason := "stop"
	switch {
	case hasToolCalls:
		reason = "tool_calls"
	case resp.Status == "incomplete" && resp.IncompleteDetails != nil && resp.IncompleteDetails.Reason == "max_output_tokens":
		reason = "length"
	case resp.Status == "incomplete":
		reason = "content_filter"
	}
	return &reason
}
