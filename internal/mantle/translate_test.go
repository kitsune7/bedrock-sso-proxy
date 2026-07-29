package mantle

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"

	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"
)

func gpt55() models.Resolved {
	return models.Resolved{
		ID:         "openai.gpt-5.5",
		Backend:    models.BackendMantleResponses,
		NoSampling: true,
		Reasoning:  true,
	}
}

func rawJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func TestTranslateRequest_ParameterAdaptation(t *testing.T) {
	temp := float32(0.7)
	req := &openai.ChatCompletionRequest{
		Model:       "gpt-5.5",
		MaxTokens:   aws.Int32(50),
		Temperature: &temp,
		Messages: []openai.Message{
			{Role: "user", Content: rawJSON(t, "hi")},
		},
	}

	out, err := TranslateRequest(req, gpt55())
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}

	// max_tokens becomes max_output_tokens, raised to the reasoning floor.
	if out.MaxOutputTokens == nil || *out.MaxOutputTokens != models.ReasoningMinOutputTokens {
		t.Errorf("MaxOutputTokens = %v, want %d", out.MaxOutputTokens, models.ReasoningMinOutputTokens)
	}
	// GPT-5 rejects sampling params, so they must not be forwarded.
	if out.Temperature != nil {
		t.Errorf("Temperature = %v, want nil", *out.Temperature)
	}
	if out.Model != "openai.gpt-5.5" {
		t.Errorf("Model = %q, want the resolved Bedrock ID", out.Model)
	}
}

func TestTranslateRequest_SystemBecomesInstructions(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Messages: []openai.Message{
			{Role: "system", Content: rawJSON(t, "be terse")},
			{Role: "developer", Content: rawJSON(t, "no preamble")},
			{Role: "user", Content: rawJSON(t, "hello")},
		},
	}

	out, err := TranslateRequest(req, gpt55())
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}

	// Responses has no system role — both system and developer messages fold
	// into the top-level instructions field, and must not appear in input.
	if !strings.Contains(out.Instructions, "be terse") || !strings.Contains(out.Instructions, "no preamble") {
		t.Errorf("Instructions = %q, want both system messages", out.Instructions)
	}
	if len(out.Input) != 1 {
		t.Fatalf("len(Input) = %d, want 1 (only the user message)", len(out.Input))
	}
	if out.Input[0].Role != "user" {
		t.Errorf("Input[0].Role = %q, want user", out.Input[0].Role)
	}
	if len(out.Input[0].Content) != 1 || out.Input[0].Content[0].Type != "input_text" {
		t.Errorf("Input[0].Content = %+v, want one input_text part", out.Input[0].Content)
	}
}

func TestTranslateRequest_ToolCallRoundTrip(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Messages: []openai.Message{
			{Role: "user", Content: rawJSON(t, "weather?")},
			{
				Role: "assistant",
				ToolCalls: []openai.ToolCall{{
					ID:       "call_abc",
					Type:     "function",
					Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city":"Boston"}`},
				}},
			},
			{Role: "tool", ToolCallID: "call_abc", Content: rawJSON(t, "62F")},
		},
		Tools: []openai.Tool{{
			Type: "function",
			Function: openai.FunctionDef{
				Name:        "get_weather",
				Description: "look up weather",
				Parameters:  rawJSON(t, map[string]any{"type": "object"}),
			},
		}},
	}

	out, err := TranslateRequest(req, gpt55())
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}

	// A tool call and its result are siblings in the flat input list, not
	// fields of the messages around them.
	wantTypes := []string{"message", "function_call", "function_call_output"}
	if len(out.Input) != len(wantTypes) {
		t.Fatalf("len(Input) = %d, want %d: %+v", len(out.Input), len(wantTypes), out.Input)
	}
	for i, want := range wantTypes {
		if out.Input[i].Type != want {
			t.Errorf("Input[%d].Type = %q, want %q", i, out.Input[i].Type, want)
		}
	}
	if out.Input[1].CallID != "call_abc" || out.Input[1].Name != "get_weather" {
		t.Errorf("function_call = %+v, want call_abc/get_weather", out.Input[1])
	}
	// The output item must carry the same call_id or the model cannot pair them.
	if out.Input[2].CallID != "call_abc" || out.Input[2].Output != "62F" {
		t.Errorf("function_call_output = %+v, want call_abc/62F", out.Input[2])
	}
	// Responses inlines the function fields rather than nesting them.
	if len(out.Tools) != 1 || out.Tools[0].Name != "get_weather" || out.Tools[0].Type != "function" {
		t.Errorf("Tools = %+v, want one inlined function tool", out.Tools)
	}
}

func TestTranslateToolChoice(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"auto passes through", `"auto"`, `"auto"`},
		{"required passes through", `"required"`, `"required"`},
		// The object form drops the nested "function" wrapper.
		{"named function is flattened", `{"type":"function","function":{"name":"f"}}`, `{"name":"f","type":"function"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := translateToolChoice(json.RawMessage(tc.in))
			if err != nil {
				t.Fatalf("translateToolChoice: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("got %s, want %s", got, tc.want)
			}
		})
	}

	if _, err := translateToolChoice(json.RawMessage(`{"type":"function"}`)); err == nil {
		t.Error("expected an error for a function tool_choice with no name")
	}
	if got, _ := translateToolChoice(nil); got != nil {
		t.Errorf("nil tool_choice = %s, want nil", got)
	}
}

func TestTranslateResponse(t *testing.T) {
	resp := &Response{
		Status: "completed",
		Output: []OutputItem{
			// Reasoning items must be skipped — they have no Chat Completions
			// equivalent and must not leak into content.
			{Type: "reasoning"},
			{Type: "message", Role: "assistant", Content: []ContentPart{
				{Type: "output_text", Text: "Hello "},
				{Type: "output_text", Text: "there"},
			}},
			{Type: "function_call", CallID: "call_1", Name: "f", Arguments: `{"x":1}`},
		},
		Usage: &Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
	}

	got := TranslateResponse(resp, "gpt-5.5")

	if len(got.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(got.Choices))
	}
	choice := got.Choices[0]
	if content := choice.Message.ContentString(); content != "Hello there" {
		t.Errorf("content = %q, want %q", content, "Hello there")
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].ID != "call_1" {
		t.Errorf("ToolCalls = %+v, want one call_1", choice.Message.ToolCalls)
	}
	// Responses reports "completed" even when stopping to call a function, but
	// Chat Completions clients branch on tool_calls to decide to execute one.
	if choice.FinishReason == nil || *choice.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %v, want tool_calls", choice.FinishReason)
	}
	if got.Usage.TotalTokens != 30 || got.Usage.PromptTokens != 10 {
		t.Errorf("Usage = %+v", got.Usage)
	}
	if got.Model != "gpt-5.5" {
		t.Errorf("Model = %q, want the name the client sent", got.Model)
	}
}

func TestFinishReason(t *testing.T) {
	incomplete := func(reason string) *Response {
		r := &Response{Status: "incomplete"}
		r.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: reason}
		return r
	}

	cases := []struct {
		name         string
		resp         *Response
		hasToolCalls bool
		want         string
	}{
		{"completed", &Response{Status: "completed"}, false, "stop"},
		{"tool calls win over status", &Response{Status: "completed"}, true, "tool_calls"},
		{"truncated", incomplete("max_output_tokens"), false, "length"},
		{"filtered", incomplete("content_filter"), false, "content_filter"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := *FinishReason(tc.resp, tc.hasToolCalls); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
