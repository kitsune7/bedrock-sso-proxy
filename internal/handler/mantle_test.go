package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"

	"bedrock-sso-proxy/internal/auth"
	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/mantle"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"
)

// fakeMantle stands in for the bedrock-mantle endpoint. It records the request
// body it received and replies with the given payload.
func fakeMantle(t *testing.T, reply string, contentType string, captured *mantle.Request) aws.Config {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
				t.Errorf("decode upstream request: %v", err)
			}
		}
		w.Header().Set("Content-Type", contentType)
		w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)

	host := strings.TrimPrefix(srv.URL, "http://")
	return aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("AKIAEXAMPLE", "secret", ""),
		HTTPClient:  &hostRewriter{host: host, inner: srv.Client()},
	}
}

type hostRewriter struct {
	host  string
	inner *http.Client
}

func (c *hostRewriter) Do(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = c.host
	return c.inner.Do(req)
}

func testRegistry() *models.Registry {
	return models.NewRegistry(&config.Config{
		DefaultModel: "claude-opus-5",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	})
}

// A gpt-5.5 request must reach the Responses API and come back in the ordinary
// chat-completion shape — the client cannot tell which Bedrock API served it.
func TestChatHandler_DispatchesGPT5ToMantle(t *testing.T) {
	reply := `{
		"status": "completed",
		"output": [
			{"type": "reasoning"},
			{"type": "message", "role": "assistant",
			 "content": [{"type": "output_text", "text": "MANTLE OK"}]}
		],
		"usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
	}`
	var upstream mantle.Request
	cfg := fakeMantle(t, reply, "application/json", &upstream)

	h := NewChatHandler(auth.NewManagerFromConfig(cfg), testRegistry(), &config.Config{})

	body := `{"model":"gpt-5.5","max_tokens":30,"temperature":0.7,
	          "messages":[{"role":"system","content":"be terse"},
	                      {"role":"user","content":"ping"}]}`
	rec := httptest.NewRecorder()
	h.Handle(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}

	// Request-side adaptation: the Responses field names and the reasoning floor.
	if upstream.Model != "openai.gpt-5.5" {
		t.Errorf("upstream model = %q, want openai.gpt-5.5", upstream.Model)
	}
	if upstream.MaxOutputTokens == nil || *upstream.MaxOutputTokens != models.ReasoningMinOutputTokens {
		t.Errorf("max_output_tokens = %v, want %d", upstream.MaxOutputTokens, models.ReasoningMinOutputTokens)
	}
	if upstream.Temperature != nil {
		t.Errorf("temperature = %v, want dropped for gpt-5.5", *upstream.Temperature)
	}
	if upstream.Instructions != "be terse" {
		t.Errorf("instructions = %q", upstream.Instructions)
	}

	var got openai.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d", len(got.Choices))
	}
	if content := got.Choices[0].Message.ContentString(); content != "MANTLE OK" {
		t.Errorf("content = %q, want MANTLE OK", content)
	}
	if got.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", got.Object)
	}
	if got.Model != "gpt-5.5" {
		t.Errorf("model = %q, want the alias the client sent", got.Model)
	}
	if got.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", got.Usage)
	}
}

func TestChatHandler_MantleStreamsAsOpenAISSE(t *testing.T) {
	reply := strings.Join([]string{
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`,
		``,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"lo"}`,
		``,
		`data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`,
		``,
	}, "\n")
	cfg := fakeMantle(t, reply, "text/event-stream", nil)

	h := NewChatHandler(auth.NewManagerFromConfig(cfg), testRegistry(), &config.Config{})

	body := `{"model":"gpt-5.5","stream":true,"stream_options":{"include_usage":true},
	          "messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.Handle(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q", ct)
	}

	out := rec.Body.String()
	var text string
	var sawRole, sawFinish, sawUsage bool
	for _, line := range strings.Split(out, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", data, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("object = %q", chunk.Object)
		}
		for _, c := range chunk.Choices {
			if c.Delta.Role == "assistant" {
				sawRole = true
			}
			text += c.Delta.Content
			if c.FinishReason != nil && *c.FinishReason == "stop" {
				sawFinish = true
			}
		}
		if chunk.Usage != nil && chunk.Usage.TotalTokens == 5 {
			sawUsage = true
		}
	}

	if !sawRole {
		t.Error("missing the initial assistant role chunk")
	}
	if text != "Hello" {
		t.Errorf("streamed text = %q, want Hello", text)
	}
	if !sawFinish {
		t.Error("missing the finish_reason chunk")
	}
	if !sawUsage {
		t.Error("missing the usage chunk")
	}
	if !strings.HasSuffix(out, "data: [DONE]\n\n") {
		t.Error("stream not terminated with [DONE]")
	}
}

// Tool calls arrive as argument deltas keyed by output_index, and must be
// reassembled into the indexed tool_calls a Chat Completions client expects.
func TestChatHandler_MantleStreamsToolCalls(t *testing.T) {
	reply := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c1","name":"get_weather"}}`,
		``,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}`,
		``,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"Boston\"}"}`,
		``,
		`data: {"type":"response.completed","response":{"status":"completed"}}`,
		``,
	}, "\n")
	cfg := fakeMantle(t, reply, "text/event-stream", nil)

	h := NewChatHandler(auth.NewManagerFromConfig(cfg), testRegistry(), &config.Config{})

	body := `{"model":"gpt-5.5","stream":true,"messages":[{"role":"user","content":"weather?"}]}`
	rec := httptest.NewRecorder()
	h.Handle(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	var name, args, finish string
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", data, err)
		}
		for _, c := range chunk.Choices {
			for _, tc := range c.Delta.ToolCalls {
				if tc.Index == nil || *tc.Index != 0 {
					t.Errorf("tool call index = %v, want 0", tc.Index)
				}
				name += tc.Function.Name
				args += tc.Function.Arguments
			}
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
	}

	if name != "get_weather" {
		t.Errorf("tool name = %q", name)
	}
	if args != `{"city":"Boston"}` {
		t.Errorf("tool arguments = %q", args)
	}
	// Responses says "completed"; a Chat Completions client needs "tool_calls"
	// to know it should execute the call.
	if finish != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", finish)
	}
}

func TestOllamaHandler_DispatchesGPT5ToMantle(t *testing.T) {
	reply := `{
		"status": "completed",
		"output": [{"type": "message", "role": "assistant",
		            "content": [{"type": "output_text", "text": "OLLAMA MANTLE OK"}]}],
		"usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
	}`
	var upstream mantle.Request
	cfg := fakeMantle(t, reply, "application/json", &upstream)

	h := NewOllamaChatHandler(auth.NewManagerFromConfig(cfg), testRegistry(), &config.Config{})

	// Ollama clients append ":latest" and use options.num_predict.
	body := `{"model":"gpt-5.5:latest","stream":false,
	          "messages":[{"role":"user","content":"ping"}],
	          "options":{"num_predict":30,"temperature":0.7}}`
	rec := httptest.NewRecorder()
	h.Handle(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if upstream.Model != "openai.gpt-5.5" {
		t.Errorf("upstream model = %q, want the tag stripped and alias resolved", upstream.Model)
	}
	if upstream.MaxOutputTokens == nil || *upstream.MaxOutputTokens != models.ReasoningMinOutputTokens {
		t.Errorf("max_output_tokens = %v, want the reasoning floor", upstream.MaxOutputTokens)
	}
	if upstream.Temperature != nil {
		t.Errorf("temperature = %v, want dropped", *upstream.Temperature)
	}

	var got struct {
		Model   string `json:"model"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Done            bool   `json:"done"`
		DoneReason      string `json:"done_reason"`
		PromptEvalCount int    `json:"prompt_eval_count"`
		EvalCount       int    `json:"eval_count"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Message.Content != "OLLAMA MANTLE OK" {
		t.Errorf("content = %q", got.Message.Content)
	}
	if !got.Done || got.DoneReason != "stop" {
		t.Errorf("done = %v, reason = %q", got.Done, got.DoneReason)
	}
	if got.PromptEvalCount != 7 || got.EvalCount != 3 {
		t.Errorf("token counts = %d/%d, want 7/3", got.PromptEvalCount, got.EvalCount)
	}
	if got.Model != "gpt-5.5:latest" {
		t.Errorf("model = %q, want the name the client sent", got.Model)
	}
}

// An upstream failure must surface as an error response, not a 200 with an
// empty completion.
func TestChatHandler_MantleUpstreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"message":"model not enabled in this account"}`))
	}))
	defer srv.Close()

	cfg := aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("AKIAEXAMPLE", "secret", ""),
		HTTPClient:  &hostRewriter{host: strings.TrimPrefix(srv.URL, "http://"), inner: srv.Client()},
	}
	h := NewChatHandler(auth.NewManagerFromConfig(cfg), testRegistry(), &config.Config{})

	rec := httptest.NewRecorder()
	h.Handle(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5.5","messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	var errResp openai.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if !strings.Contains(errResp.Error.Message, "model not enabled") {
		t.Errorf("error message = %q, want the upstream detail", errResp.Error.Message)
	}
}
