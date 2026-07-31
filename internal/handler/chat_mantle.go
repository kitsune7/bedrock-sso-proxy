package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"

	"bedrock-sso-proxy/internal/mantle"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"

	"github.com/aws/aws-sdk-go-v2/aws"
)

// handleMantle serves models reachable only through the Responses API on the
// bedrock-mantle endpoint. The request and response translation differ from
// Converse, but the client-facing shape is identical — the caller cannot tell
// which Bedrock API served the request.
func (h *ChatHandler) handleMantle(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, resolved models.Resolved) {
	mreq, err := mantle.TranslateRequest(r.Context(), req, resolved)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	var resp *http.Response
	err = h.authMgr.WithRetryConfig(r.Context(), func(cfg aws.Config) error {
		var callErr error
		resp, callErr = mantle.Do(r.Context(), cfg, mreq)
		return callErr
	})
	if err != nil {
		log.Printf("Mantle Responses error: %v", err)
		writeError(w, http.StatusBadGateway, "api_error", "Bedrock API error: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if req.Stream {
		h.streamMantle(w, req, resp)
		return
	}

	var mresp mantle.Response
	if err := json.NewDecoder(resp.Body).Decode(&mresp); err != nil {
		log.Printf("Mantle response decode error: %v", err)
		writeError(w, http.StatusBadGateway, "api_error", "failed to decode Bedrock response: "+err.Error())
		return
	}
	if mresp.Error != nil {
		writeError(w, http.StatusBadGateway, "api_error", mresp.Error.Message)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(mantle.TranslateResponse(&mresp, req.Model))
}

// streamMantle converts the Responses SSE event stream into OpenAI chat
// completion chunks. Headers are only written here, after the upstream call has
// already succeeded, so an error before this point can still be a JSON body.
func (h *ChatHandler) streamMantle(w http.ResponseWriter, req *openai.ChatCompletionRequest, resp *http.Response) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	completionID := "chatcmpl-" + uuid.New().String()
	created := time.Now().Unix()
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	newChunk := func(choices []openai.ChunkChoice) *openai.ChatCompletionChunk {
		return &openai.ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: choices,
		}
	}

	send := func(chunk *openai.ChatCompletionChunk) error {
		data, err := json.Marshal(chunk)
		if err != nil {
			return fmt.Errorf("marshal SSE chunk: %w", err)
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	if err := send(newChunk([]openai.ChunkChoice{{
		Index: 0,
		Delta: openai.Delta{Role: "assistant"},
	}})); err != nil {
		log.Printf("mantle stream write error: %v", err)
		return
	}

	// Responses keys function-call argument deltas by output_index, and reveals
	// the name and call_id only on the item-added event. OpenAI chunks index
	// tool calls by their position among tool calls, so map one to the other.
	toolIndexByOutput := map[int]int{}
	var toolCount int
	var sawToolCall bool

	err := mantle.DecodeStream(resp.Body, func(event mantle.StreamEvent) error {
		switch event.Type {
		case mantle.EventOutputTextDelta:
			if event.Delta == "" {
				return nil
			}
			return send(newChunk([]openai.ChunkChoice{{
				Index: 0,
				Delta: openai.Delta{Content: event.Delta},
			}}))

		case mantle.EventOutputItemAdded:
			if event.Item == nil || event.Item.Type != "function_call" {
				return nil
			}
			idx := toolCount
			toolCount++
			toolIndexByOutput[event.OutputIndex] = idx
			sawToolCall = true
			return send(newChunk([]openai.ChunkChoice{{
				Index: 0,
				Delta: openai.Delta{ToolCalls: []openai.ToolCall{{
					Index:    &idx,
					ID:       event.Item.CallID,
					Type:     "function",
					Function: openai.FunctionCall{Name: event.Item.Name},
				}}},
			}}))

		case mantle.EventFunctionArgsDelta:
			idx, ok := toolIndexByOutput[event.OutputIndex]
			if !ok || event.Delta == "" {
				return nil
			}
			return send(newChunk([]openai.ChunkChoice{{
				Index: 0,
				Delta: openai.Delta{ToolCalls: []openai.ToolCall{{
					Index:    &idx,
					Function: openai.FunctionCall{Arguments: event.Delta},
				}}},
			}}))

		case mantle.EventCompleted, mantle.EventIncomplete:
			final := &mantle.Response{Status: "completed"}
			if event.Response != nil {
				final = event.Response
			}
			if err := send(newChunk([]openai.ChunkChoice{{
				Index:        0,
				Delta:        openai.Delta{},
				FinishReason: mantle.FinishReason(final, sawToolCall),
			}})); err != nil {
				return err
			}
			if includeUsage && final.Usage != nil {
				chunk := newChunk([]openai.ChunkChoice{})
				chunk.Usage = &openai.Usage{
					PromptTokens:     final.Usage.InputTokens,
					CompletionTokens: final.Usage.OutputTokens,
					TotalTokens:      final.Usage.TotalTokens,
				}
				return send(chunk)
			}
			return nil

		case mantle.EventFailed:
			// The stream is already committed with a 200, so an upstream failure
			// can only be reported in the log and by terminating the stream.
			if event.Response != nil && event.Response.Error != nil {
				log.Printf("mantle stream failed: %s", event.Response.Error.Message)
			} else {
				log.Print("mantle stream failed with no error detail")
			}
			return nil
		}
		return nil
	})
	if err != nil {
		log.Printf("mantle stream error: %v", err)
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
