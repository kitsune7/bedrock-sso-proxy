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

	"bedrock-sso-proxy/internal/auth"
	"bedrock-sso-proxy/internal/bedrock"
	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/ollama"
	"bedrock-sso-proxy/internal/openai"
)

// OllamaChatHandler handles POST /api/chat (Ollama-compatible).
type OllamaChatHandler struct {
	authMgr  *auth.AuthManager
	registry *models.Registry
	cfg      *config.Config
}

func NewOllamaChatHandler(authMgr *auth.AuthManager, registry *models.Registry, cfg *config.Config) *OllamaChatHandler {
	return &OllamaChatHandler{authMgr: authMgr, registry: registry, cfg: cfg}
}

func (h *OllamaChatHandler) Handle(w http.ResponseWriter, r *http.Request) {
	var req ollama.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOllamaError(w, http.StatusBadRequest, "failed to parse request body: "+err.Error())
		return
	}

	modelID := h.registry.ResolveModelID(req.Model)
	if h.cfg.Verbose {
		log.Printf("ollama model: %s -> %s", req.Model, modelID)
	}

	openaiReq, err := ollamaToOpenAI(&req)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.StreamEnabled() {
		h.handleStream(w, r, &req, openaiReq, modelID)
	} else {
		h.handleNonStream(w, r, &req, openaiReq, modelID)
	}
}

func (h *OllamaChatHandler) handleNonStream(w http.ResponseWriter, r *http.Request, req *ollama.ChatRequest, openaiReq *openai.ChatCompletionRequest, modelID string) {
	input, err := bedrock.TranslateRequest(openaiReq, modelID)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}

	start := time.Now()
	var output *bedrockruntime.ConverseOutput
	err = h.authMgr.WithRetry(r.Context(), func(client *bedrockruntime.Client) error {
		var callErr error
		output, callErr = bedrock.Converse(r.Context(), client, input)
		return callErr
	})
	if err != nil {
		log.Printf("Bedrock Converse error: %v", err)
		writeOllamaError(w, http.StatusBadGateway, "Bedrock API error: "+err.Error())
		return
	}

	resp := ollama.ChatResponse{
		Model:     req.Model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Message:   ollama.Message{Role: "assistant"},
		Done:      true,
	}

	if msgOutput, ok := output.Output.(*types.ConverseOutputMemberMessage); ok {
		text, toolCalls := extractOllamaContent(msgOutput.Value.Content)
		resp.Message.Content = text
		resp.Message.ToolCalls = toolCalls
	}

	resp.DoneReason = ollamaStopReason(output.StopReason)
	resp.TotalDuration = time.Since(start).Nanoseconds()
	if output.Usage != nil {
		resp.PromptEvalCount = int(aws.ToInt32(output.Usage.InputTokens))
		resp.EvalCount = int(aws.ToInt32(output.Usage.OutputTokens))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *OllamaChatHandler) handleStream(w http.ResponseWriter, r *http.Request, req *ollama.ChatRequest, openaiReq *openai.ChatCompletionRequest, modelID string) {
	input, err := bedrock.TranslateStreamRequest(openaiReq, modelID)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}

	start := time.Now()
	var streamOutput *bedrockruntime.ConverseStreamOutput
	err = h.authMgr.WithRetry(r.Context(), func(client *bedrockruntime.Client) error {
		var callErr error
		streamOutput, callErr = bedrock.ConverseStream(r.Context(), client, input)
		return callErr
	})
	if err != nil {
		log.Printf("Bedrock ConverseStream error: %v", err)
		writeOllamaError(w, http.StatusBadGateway, "Bedrock API error: "+err.Error())
		return
	}

	// Ollama streams are newline-delimited JSON, not SSE.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOllamaError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	stream := streamOutput.GetStream()
	defer stream.Close()

	var (
		doneReason      string
		promptTokens    int
		completionTokens int

		// Buffer tool-use arguments across ContentBlockDelta events; emit on stop.
		toolAccum   = map[int]*pendingToolCall{}
		toolByIndex []*pendingToolCall
	)

	enc := json.NewEncoder(w)

	for event := range stream.Events() {
		switch v := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if v.Value.Start != nil {
				if tu, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					idx := int(aws.ToInt32(v.Value.ContentBlockIndex))
					pc := &pendingToolCall{
						Name: aws.ToString(tu.Value.Name),
					}
					toolAccum[idx] = pc
					toolByIndex = append(toolByIndex, pc)
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if v.Value.Delta == nil {
				continue
			}
			switch d := v.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				frame := ollama.ChatResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
					Message:   ollama.Message{Role: "assistant", Content: d.Value},
					Done:      false,
				}
				if err := enc.Encode(frame); err != nil {
					log.Printf("ollama stream encode error: %v", err)
					return
				}
				flusher.Flush()

			case *types.ContentBlockDeltaMemberToolUse:
				idx := int(aws.ToInt32(v.Value.ContentBlockIndex))
				if pc, ok := toolAccum[idx]; ok {
					pc.ArgsRaw += aws.ToString(d.Value.Input)
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			doneReason = ollamaStopReason(v.Value.StopReason)

		case *types.ConverseStreamOutputMemberMetadata:
			if v.Value.Usage != nil {
				promptTokens = int(aws.ToInt32(v.Value.Usage.InputTokens))
				completionTokens = int(aws.ToInt32(v.Value.Usage.OutputTokens))
			}
		}
	}

	if err := stream.Err(); err != nil {
		log.Printf("ollama stream error: %v", err)
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

	if len(toolByIndex) > 0 {
		for _, pc := range toolByIndex {
			var argsMap map[string]any
			if pc.ArgsRaw != "" {
				if err := json.Unmarshal([]byte(pc.ArgsRaw), &argsMap); err != nil {
					argsMap = map[string]any{"raw": pc.ArgsRaw}
				}
			}
			if argsMap == nil {
				argsMap = map[string]any{}
			}
			final.Message.ToolCalls = append(final.Message.ToolCalls, ollama.ToolCall{
				Function: ollama.ToolCallFunction{
					Name:      pc.Name,
					Arguments: argsMap,
				},
			})
		}
	}

	if err := enc.Encode(final); err != nil {
		log.Printf("ollama final frame encode error: %v", err)
	}
	flusher.Flush()
}

type pendingToolCall struct {
	Name    string
	ArgsRaw string
}

// OllamaTagsHandler handles GET /api/tags — the Ollama "list installed models" endpoint.
type OllamaTagsHandler struct {
	registry *models.Registry
}

func NewOllamaTagsHandler(registry *models.Registry) *OllamaTagsHandler {
	return &OllamaTagsHandler{registry: registry}
}

func (h *OllamaTagsHandler) Handle(w http.ResponseWriter, r *http.Request) {
	list := h.registry.ListModels()
	modified := time.Now().UTC().Format(time.RFC3339Nano)

	resp := ollama.TagsResponse{
		Models: make([]ollama.TagModel, 0, len(list.Data)),
	}
	for _, m := range list.Data {
		// Ollama clients expect a "name:tag" form. Using ":latest" keeps the
		// round-trip consistent — clients will echo it back on /api/chat.
		tagged := m.ID + ":latest"
		resp.Models = append(resp.Models, ollama.TagModel{
			Name:       tagged,
			Model:      tagged,
			ModifiedAt: modified,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ollamaToOpenAI converts an Ollama chat request to the internal OpenAI-shaped
// request so we can reuse bedrock.TranslateRequest / TranslateStreamRequest.
func ollamaToOpenAI(req *ollama.ChatRequest) (*openai.ChatCompletionRequest, error) {
	out := &openai.ChatCompletionRequest{
		Model:    req.Model,
		Messages: make([]openai.Message, 0, len(req.Messages)),
	}

	// Ollama's wire format has no tool-call IDs — assistant tool_calls and the
	// matching tool-result messages are paired implicitly by order. Bedrock
	// requires a non-empty toolUseId on both sides, so synthesize a FIFO queue
	// of deterministic IDs and pop one per tool-result message.
	var pendingToolIDs []string
	for msgIdx, m := range req.Messages {
		contentJSON, err := json.Marshal(m.Content)
		if err != nil {
			return nil, fmt.Errorf("encode message content: %w", err)
		}
		msg := openai.Message{
			Role:    m.Role,
			Content: contentJSON,
		}
		for callIdx, tc := range m.ToolCalls {
			argsJSON, err := json.Marshal(tc.Function.Arguments)
			if err != nil {
				return nil, fmt.Errorf("encode tool-call arguments: %w", err)
			}
			id := fmt.Sprintf("call_%d_%d", msgIdx, callIdx)
			pendingToolIDs = append(pendingToolIDs, id)
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:   id,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: string(argsJSON),
				},
			})
		}
		if m.Role == "tool" && msg.ToolCallID == "" && len(pendingToolIDs) > 0 {
			msg.ToolCallID = pendingToolIDs[0]
			pendingToolIDs = pendingToolIDs[1:]
		}
		out.Messages = append(out.Messages, msg)
	}

	if req.Options != nil {
		if req.Options.NumPredict != nil && *req.Options.NumPredict > 0 {
			out.MaxTokens = req.Options.NumPredict
		}
		out.Temperature = req.Options.Temperature
		out.TopP = req.Options.TopP
		if len(req.Options.Stop) > 0 {
			out.Stop = openai.StringOrStrings(req.Options.Stop)
		}
	}

	for _, t := range req.Tools {
		if t.Type == "" {
			t.Type = "function"
		}
		out.Tools = append(out.Tools, openai.Tool{
			Type: t.Type,
			Function: openai.FunctionDef{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			},
		})
	}

	return out, nil
}

func extractOllamaContent(blocks []types.ContentBlock) (string, []ollama.ToolCall) {
	var text string
	var toolCalls []ollama.ToolCall

	for _, block := range blocks {
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			text += b.Value
		case *types.ContentBlockMemberToolUse:
			var args map[string]any
			// The document.Interface value unmarshals into a map via its own marshaler.
			if raw, err := b.Value.Input.MarshalSmithyDocument(); err == nil {
				_ = json.Unmarshal(raw, &args)
			}
			if args == nil {
				args = map[string]any{}
			}
			toolCalls = append(toolCalls, ollama.ToolCall{
				Function: ollama.ToolCallFunction{
					Name:      aws.ToString(b.Value.Name),
					Arguments: args,
				},
			})
		}
	}

	return text, toolCalls
}

func ollamaStopReason(reason types.StopReason) string {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return "stop"
	case types.StopReasonToolUse:
		return "tool_calls"
	case types.StopReasonMaxTokens:
		return "length"
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return "content_filter"
	default:
		return "stop"
	}
}

func writeOllamaError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ollama.ErrorResponse{Error: message})
}
