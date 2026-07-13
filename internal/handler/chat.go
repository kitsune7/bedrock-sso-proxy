package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/google/uuid"

	"bedrock-sso-proxy/internal/auth"
	"bedrock-sso-proxy/internal/bedrock"
	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"
)

// ChatHandler handles POST /v1/chat/completions.
type ChatHandler struct {
	authMgr  *auth.AuthManager
	registry *models.Registry
	cfg      *config.Config
}

func NewChatHandler(authMgr *auth.AuthManager, registry *models.Registry, cfg *config.Config) *ChatHandler {
	return &ChatHandler{
		authMgr:  authMgr,
		registry: registry,
		cfg:      cfg,
	}
}

func (h *ChatHandler) Handle(w http.ResponseWriter, r *http.Request) {
	var req openai.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "failed to parse request body: "+err.Error())
		return
	}

	modelID := h.registry.ResolveModelID(req.Model)
	if h.cfg.Verbose {
		log.Printf("model: %s -> %s", req.Model, modelID)
	}

	if req.Stream {
		h.handleStream(w, r, &req, modelID)
	} else {
		h.handleNonStream(w, r, &req, modelID)
	}
}

func (h *ChatHandler) handleNonStream(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, modelID string) {
	input, err := bedrock.TranslateRequest(req, modelID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	var output *bedrockruntime.ConverseOutput
	err = h.authMgr.WithRetry(r.Context(), func(client *bedrockruntime.Client) error {
		var callErr error
		output, callErr = bedrock.Converse(r.Context(), client, input)
		return callErr
	})
	if err != nil {
		log.Printf("Bedrock Converse error: %v", err)
		writeError(w, http.StatusBadGateway, "api_error", "Bedrock API error: "+err.Error())
		return
	}

	resp := bedrock.TranslateResponse(output, req.Model)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *ChatHandler) handleStream(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, modelID string) {
	input, err := bedrock.TranslateStreamRequest(req, modelID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	var streamOutput *bedrockruntime.ConverseStreamOutput
	err = h.authMgr.WithRetry(r.Context(), func(client *bedrockruntime.Client) error {
		var callErr error
		streamOutput, callErr = bedrock.ConverseStream(r.Context(), client, input)
		return callErr
	})
	if err != nil {
		log.Printf("Bedrock ConverseStream error: %v", err)
		writeError(w, http.StatusBadGateway, "api_error", "Bedrock API error: "+err.Error())
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming not supported")
		return
	}

	completionID := "chatcmpl-" + uuid.New().String()
	created := time.Now().Unix()
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	stream := streamOutput.GetStream()
	defer stream.Close()

	// Track tool call state for streaming
	toolCallIndex := -1

	for event := range stream.Events() {
		var chunk *openai.ChatCompletionChunk

		switch v := event.(type) {
		case *types.ConverseStreamOutputMemberMessageStart:
			_ = v // Role info
			chunk = &openai.ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []openai.ChunkChoice{{
					Index: 0,
					Delta: openai.Delta{Role: "assistant"},
				}},
			}

		case *types.ConverseStreamOutputMemberContentBlockStart:
			if toolUse := v.Value.Start; toolUse != nil {
				if tu, ok := toolUse.(*types.ContentBlockStartMemberToolUse); ok {
					toolCallIndex++
					idx := toolCallIndex
					chunk = &openai.ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   req.Model,
						Choices: []openai.ChunkChoice{{
							Index: 0,
							Delta: openai.Delta{
								ToolCalls: []openai.ToolCall{{
									Index: &idx,
									ID:    aws.ToString(tu.Value.ToolUseId),
									Type:  "function",
									Function: openai.FunctionCall{
										Name:      aws.ToString(tu.Value.Name),
										Arguments: "",
									},
								}},
							},
						}},
					}
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if v.Value.Delta != nil {
				switch d := v.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					chunk = &openai.ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   req.Model,
						Choices: []openai.ChunkChoice{{
							Index: 0,
							Delta: openai.Delta{Content: d.Value},
						}},
					}

				case *types.ContentBlockDeltaMemberToolUse:
					idx := toolCallIndex
					chunk = &openai.ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   req.Model,
						Choices: []openai.ChunkChoice{{
							Index: 0,
							Delta: openai.Delta{
								ToolCalls: []openai.ToolCall{{
									Index: &idx,
									Function: openai.FunctionCall{
										Arguments: aws.ToString(d.Value.Input),
									},
								}},
							},
						}},
					}
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockStop:
			// No chunk needed for content block stop

		case *types.ConverseStreamOutputMemberMessageStop:
			reason := mapStopReason(v.Value.StopReason)
			chunk = &openai.ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []openai.ChunkChoice{{
					Index:        0,
					Delta:        openai.Delta{},
					FinishReason: reason,
				}},
			}

		case *types.ConverseStreamOutputMemberMetadata:
			if includeUsage && v.Value.Usage != nil {
				inputTokens := int(aws.ToInt32(v.Value.Usage.InputTokens))
				outputTokens := int(aws.ToInt32(v.Value.Usage.OutputTokens))
				chunk = &openai.ChatCompletionChunk{
					ID:      completionID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []openai.ChunkChoice{},
					Usage: &openai.Usage{
						PromptTokens:     inputTokens,
						CompletionTokens: outputTokens,
						TotalTokens:      inputTokens + outputTokens,
					},
				}
			}
		}

		if chunk != nil {
			data, err := json.Marshal(chunk)
			if err != nil {
				log.Printf("failed to marshal SSE chunk: %v", err)
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}

	if err := stream.Err(); err != nil {
		log.Printf("stream error: %v", err)
	}

	// Terminate SSE stream
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func mapStopReason(reason types.StopReason) *string {
	var mapped string
	switch reason {
	case types.StopReasonEndTurn:
		mapped = "stop"
	case types.StopReasonToolUse:
		mapped = "tool_calls"
	case types.StopReasonMaxTokens:
		mapped = "length"
	case types.StopReasonStopSequence:
		mapped = "stop"
	case types.StopReasonContentFiltered:
		mapped = "content_filter"
	case types.StopReasonGuardrailIntervened:
		mapped = "content_filter"
	default:
		mapped = "stop"
	}
	return &mapped
}

func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(openai.ErrorResponse{
		Error: openai.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}
