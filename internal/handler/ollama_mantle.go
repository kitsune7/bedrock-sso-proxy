package handler

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"bedrock-sso-proxy/internal/mantle"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/ollama"
	"bedrock-sso-proxy/internal/openai"

	"github.com/aws/aws-sdk-go-v2/aws"
)

// handleMantle serves Responses-API models on the Ollama endpoint. The Ollama
// request has already been converted to the internal OpenAI shape, so this only
// has to translate the response back to Ollama's frames.
func (h *OllamaChatHandler) handleMantle(w http.ResponseWriter, r *http.Request, req *ollama.ChatRequest, openaiReq *openai.ChatCompletionRequest, resolved models.Resolved) {
	mreq, err := mantle.TranslateRequest(r.Context(), openaiReq, resolved)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	mreq.Stream = req.StreamEnabled()

	start := time.Now()
	var resp *http.Response
	err = h.authMgr.WithRetryConfig(r.Context(), func(cfg aws.Config) error {
		var callErr error
		resp, callErr = mantle.Do(r.Context(), cfg, mreq)
		return callErr
	})
	if err != nil {
		log.Printf("Mantle Responses error: %v", err)
		writeOllamaError(w, http.StatusBadGateway, "Bedrock API error: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if req.StreamEnabled() {
		h.streamMantle(w, req, resp, start)
		return
	}

	var mresp mantle.Response
	if err := json.NewDecoder(resp.Body).Decode(&mresp); err != nil {
		log.Printf("Mantle response decode error: %v", err)
		writeOllamaError(w, http.StatusBadGateway, "failed to decode Bedrock response: "+err.Error())
		return
	}
	if mresp.Error != nil {
		writeOllamaError(w, http.StatusBadGateway, mresp.Error.Message)
		return
	}

	// Reuse the OpenAI-shaped translation, then project it onto Ollama's frame.
	// The two surfaces carry the same information in different envelopes.
	converted := mantle.TranslateResponse(&mresp, req.Model)
	out := ollama.ChatResponse{
		Model:         req.Model,
		CreatedAt:     time.Now().UTC().Format(time.RFC3339Nano),
		Message:       ollama.Message{Role: "assistant"},
		Done:          true,
		TotalDuration: time.Since(start).Nanoseconds(),
	}
	if len(converted.Choices) > 0 {
		choice := converted.Choices[0]
		out.Message.Content = choice.Message.ContentString()
		out.Message.ToolCalls = toOllamaToolCalls(choice.Message.ToolCalls)
		if choice.FinishReason != nil {
			out.DoneReason = *choice.FinishReason
		}
	}
	out.PromptEvalCount = converted.Usage.PromptTokens
	out.EvalCount = converted.Usage.CompletionTokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// streamMantle converts Responses SSE events into Ollama's NDJSON frames. Like
// the Converse path, tool calls are buffered and emitted on the final frame —
// Ollama has no incremental tool-call representation.
func (h *OllamaChatHandler) streamMantle(w http.ResponseWriter, req *ollama.ChatRequest, resp *http.Response, start time.Time) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOllamaError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")

	enc := json.NewEncoder(w)

	var (
		doneReason       string
		promptTokens     int
		completionTokens int
		toolAccum        = map[int]*pendingToolCall{}
		toolByIndex      []*pendingToolCall
	)

	err := mantle.DecodeStream(resp.Body, func(event mantle.StreamEvent) error {
		switch event.Type {
		case mantle.EventOutputTextDelta:
			if event.Delta == "" {
				return nil
			}
			frame := ollama.ChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
				Message:   ollama.Message{Role: "assistant", Content: event.Delta},
				Done:      false,
			}
			if err := enc.Encode(frame); err != nil {
				return err
			}
			flusher.Flush()

		case mantle.EventOutputItemAdded:
			if event.Item == nil || event.Item.Type != "function_call" {
				return nil
			}
			pc := &pendingToolCall{Name: event.Item.Name}
			toolAccum[event.OutputIndex] = pc
			toolByIndex = append(toolByIndex, pc)

		case mantle.EventFunctionArgsDelta:
			if pc, ok := toolAccum[event.OutputIndex]; ok {
				pc.ArgsRaw += event.Delta
			}

		case mantle.EventCompleted, mantle.EventIncomplete:
			final := &mantle.Response{Status: "completed"}
			if event.Response != nil {
				final = event.Response
			}
			doneReason = *mantle.FinishReason(final, len(toolByIndex) > 0)
			if final.Usage != nil {
				promptTokens = final.Usage.InputTokens
				completionTokens = final.Usage.OutputTokens
			}

		case mantle.EventFailed:
			if event.Response != nil && event.Response.Error != nil {
				log.Printf("mantle stream failed: %s", event.Response.Error.Message)
			} else {
				log.Print("mantle stream failed with no error detail")
			}
		}
		return nil
	})
	if err != nil {
		log.Printf("mantle ollama stream error: %v", err)
	}

	final := ollama.ChatResponse{
		Model:           req.Model,
		CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
		Message:         ollama.Message{Role: "assistant", Content: ""},
		Done:            true,
		DoneReason:      doneReason,
		TotalDuration:   time.Since(start).Nanoseconds(),
		PromptEvalCount: promptTokens,
		EvalCount:       completionTokens,
	}
	for _, pc := range toolByIndex {
		final.Message.ToolCalls = append(final.Message.ToolCalls, ollama.ToolCall{
			Function: ollama.ToolCallFunction{
				Name:      pc.Name,
				Arguments: decodeToolArgs(pc.ArgsRaw),
			},
		})
	}

	if err := enc.Encode(final); err != nil {
		log.Printf("ollama final frame encode error: %v", err)
	}
	flusher.Flush()
}

// toOllamaToolCalls converts OpenAI tool calls, whose arguments are a JSON
// string, to Ollama's form, where they are a decoded object.
func toOllamaToolCalls(calls []openai.ToolCall) []ollama.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ollama.ToolCall, 0, len(calls))
	for _, tc := range calls {
		out = append(out, ollama.ToolCall{
			Function: ollama.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: decodeToolArgs(tc.Function.Arguments),
			},
		})
	}
	return out
}

// decodeToolArgs parses a JSON arguments string into the map Ollama expects,
// falling back to a raw wrapper so malformed arguments still reach the client
// rather than being dropped.
func decodeToolArgs(raw string) map[string]any {
	if raw == "" {
		return map[string]any{}
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil || args == nil {
		return map[string]any{"raw": raw}
	}
	return args
}
