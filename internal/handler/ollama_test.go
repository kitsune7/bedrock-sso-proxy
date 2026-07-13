package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/ollama"
)

func TestOllamaToOpenAI_MessagesAndOptions(t *testing.T) {
	numPredict := int32(256)
	temp := float32(0.4)
	topP := float32(0.9)

	in := &ollama.ChatRequest{
		Model: "claude-sonnet-4-6",
		Messages: []ollama.Message{
			{Role: "system", Content: "be concise"},
			{Role: "user", Content: "hi"},
		},
		Options: &ollama.Options{
			NumPredict:  &numPredict,
			Temperature: &temp,
			TopP:        &topP,
			Stop:        []string{"<END>"},
		},
	}

	out, err := ollamaToOpenAI(in)
	if err != nil {
		t.Fatalf("ollamaToOpenAI: %v", err)
	}

	if out.Model != "claude-sonnet-4-6" {
		t.Errorf("model: got %q", out.Model)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("messages: got %d, want 2", len(out.Messages))
	}
	if got := out.Messages[1].ContentString(); got != "hi" {
		t.Errorf("user content: got %q, want %q", got, "hi")
	}
	if out.MaxTokens == nil || *out.MaxTokens != 256 {
		t.Errorf("max_tokens: got %v, want 256", out.MaxTokens)
	}
	if out.Temperature == nil || *out.Temperature != 0.4 {
		t.Errorf("temperature: got %v, want 0.4", out.Temperature)
	}
	if out.TopP == nil || *out.TopP != 0.9 {
		t.Errorf("top_p: got %v, want 0.9", out.TopP)
	}
	if len(out.Stop) != 1 || out.Stop[0] != "<END>" {
		t.Errorf("stop: got %v", out.Stop)
	}
}

func TestOllamaToOpenAI_ToolsAndToolCalls(t *testing.T) {
	params := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)

	in := &ollama.ChatRequest{
		Model: "claude-sonnet-4-6",
		Messages: []ollama.Message{
			{
				Role:    "assistant",
				Content: "",
				ToolCalls: []ollama.ToolCall{{
					Function: ollama.ToolCallFunction{
						Name:      "search",
						Arguments: map[string]any{"q": "llamas"},
					},
				}},
			},
		},
		Tools: []ollama.Tool{{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        "search",
				Description: "Search the web",
				Parameters:  params,
			},
		}},
	}

	out, err := ollamaToOpenAI(in)
	if err != nil {
		t.Fatalf("ollamaToOpenAI: %v", err)
	}

	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "search" {
		t.Fatalf("tool def not translated: %+v", out.Tools)
	}
	if len(out.Messages[0].ToolCalls) != 1 {
		t.Fatalf("tool call not translated: %+v", out.Messages[0].ToolCalls)
	}
	tc := out.Messages[0].ToolCalls[0]
	if tc.Type != "function" || tc.Function.Name != "search" {
		t.Errorf("tool call shape: %+v", tc)
	}
	if !strings.Contains(tc.Function.Arguments, `"q":"llamas"`) {
		t.Errorf("tool call args: got %q", tc.Function.Arguments)
	}
}

func TestOllamaToOpenAI_SynthesizesToolUseIDs(t *testing.T) {
	in := &ollama.ChatRequest{
		Model: "claude-opus-4-7",
		Messages: []ollama.Message{
			{Role: "user", Content: "do the thing"},
			{Role: "assistant", Content: "", ToolCalls: []ollama.ToolCall{
				{Function: ollama.ToolCallFunction{Name: "search", Arguments: map[string]any{"q": "a"}}},
				{Function: ollama.ToolCallFunction{Name: "lookup", Arguments: map[string]any{"id": 1}}},
			}},
			{Role: "tool", Content: "result A"},
			{Role: "tool", Content: "result B"},
		},
	}

	out, err := ollamaToOpenAI(in)
	if err != nil {
		t.Fatalf("ollamaToOpenAI: %v", err)
	}

	assistant := out.Messages[1]
	if len(assistant.ToolCalls) != 2 {
		t.Fatalf("tool calls: got %d, want 2", len(assistant.ToolCalls))
	}
	id0, id1 := assistant.ToolCalls[0].ID, assistant.ToolCalls[1].ID
	if id0 == "" || id1 == "" {
		t.Fatalf("tool-call IDs must be non-empty: %q, %q", id0, id1)
	}
	if id0 == id1 {
		t.Fatalf("tool-call IDs must be distinct: %q", id0)
	}

	if out.Messages[2].ToolCallID != id0 {
		t.Errorf("tool-result 0: got id %q, want %q", out.Messages[2].ToolCallID, id0)
	}
	if out.Messages[3].ToolCallID != id1 {
		t.Errorf("tool-result 1: got id %q, want %q", out.Messages[3].ToolCallID, id1)
	}
}

func TestOllamaTagsHandler_ListsRegistryModels(t *testing.T) {
	reg := models.NewRegistry(&config.Config{DefaultModel: "claude-sonnet-4-6"})
	h := NewOllamaTagsHandler(reg)

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	rec := httptest.NewRecorder()
	h.Handle(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d", rec.Code)
	}
	var body ollama.TagsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Models) == 0 {
		t.Fatal("expected at least one model")
	}
	// Sanity: confirm a known alias is present and populated.
	var found bool
	for _, m := range body.Models {
		if m.Name == "claude-sonnet-4-6:latest" && m.Model == "claude-sonnet-4-6:latest" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected claude-sonnet-4-6:latest in tags, got %+v", body.Models)
	}
}

func TestOllamaChatHandler_RejectsBadJSON(t *testing.T) {
	reg := models.NewRegistry(&config.Config{DefaultModel: "claude-sonnet-4-6"})
	h := NewOllamaChatHandler(nil, reg, &config.Config{})

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader("{not json"))
	rec := httptest.NewRecorder()
	h.Handle(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d, want 400", rec.Code)
	}
	var errBody ollama.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errBody); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if errBody.Error == "" {
		t.Errorf("expected error message")
	}
}
