package bedrock

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/google/uuid"

	"bedrock-sso-proxy/internal/openai"
)

// modelsWithoutTemperature lists model-ID substrings for which Bedrock rejects
// the `temperature` inference parameter. Match is case-insensitive and
// substring-based so it covers alias, bare ID, and region-prefixed forms.
var modelsWithoutTemperature = []string{
	"claude-opus-4-8",
	"claude-opus-4-7",
	"claude-sonnet-5",
}

// TranslateRequest converts an OpenAI ChatCompletionRequest into Bedrock ConverseInput.
func TranslateRequest(req *openai.ChatCompletionRequest, modelID string) (*bedrockruntime.ConverseInput, error) {
	system, messages, err := translateMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(modelID),
		Messages: messages,
	}

	if len(system) > 0 {
		input.System = system
	}

	if inferCfg := translateInferenceConfig(req, modelID); inferCfg != nil {
		input.InferenceConfig = inferCfg
	}

	if toolCfg, err := translateToolConfig(req); err != nil {
		return nil, err
	} else if toolCfg != nil {
		input.ToolConfig = toolCfg
	}

	return input, nil
}

// TranslateStreamRequest converts an OpenAI ChatCompletionRequest into Bedrock ConverseStreamInput.
func TranslateStreamRequest(req *openai.ChatCompletionRequest, modelID string) (*bedrockruntime.ConverseStreamInput, error) {
	system, messages, err := translateMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(modelID),
		Messages: messages,
	}

	if len(system) > 0 {
		input.System = system
	}

	if inferCfg := translateInferenceConfig(req, modelID); inferCfg != nil {
		input.InferenceConfig = inferCfg
	}

	if toolCfg, err := translateToolConfig(req); err != nil {
		return nil, err
	} else if toolCfg != nil {
		input.ToolConfig = toolCfg
	}

	return input, nil
}

// TranslateResponse converts a Bedrock ConverseOutput to an OpenAI ChatCompletionResponse.
func TranslateResponse(output *bedrockruntime.ConverseOutput, model string) *openai.ChatCompletionResponse {
	resp := &openai.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
	}

	choice := openai.Choice{Index: 0}

	// Extract message content
	if msgOutput, ok := output.Output.(*types.ConverseOutputMemberMessage); ok {
		msg := openai.Message{Role: "assistant"}
		var textParts []string
		var toolCalls []openai.ToolCall

		for _, block := range msgOutput.Value.Content {
			switch b := block.(type) {
			case *types.ContentBlockMemberText:
				textParts = append(textParts, b.Value)
			case *types.ContentBlockMemberToolUse:
				argsJSON, _ := json.Marshal(b.Value.Input)
				toolCalls = append(toolCalls, openai.ToolCall{
					ID:   aws.ToString(b.Value.ToolUseId),
					Type: "function",
					Function: openai.FunctionCall{
						Name:      aws.ToString(b.Value.Name),
						Arguments: string(argsJSON),
					},
				})
			}
		}

		if len(textParts) > 0 {
			content := ""
			for _, p := range textParts {
				content += p
			}
			contentJSON, _ := json.Marshal(content)
			msg.Content = contentJSON
		}
		if len(toolCalls) > 0 {
			msg.ToolCalls = toolCalls
		}
		choice.Message = msg
	}

	// Map stop reason
	choice.FinishReason = mapStopReason(output.StopReason)

	resp.Choices = []openai.Choice{choice}

	// Map usage
	if output.Usage != nil {
		inputTokens := int(aws.ToInt32(output.Usage.InputTokens))
		outputTokens := int(aws.ToInt32(output.Usage.OutputTokens))
		resp.Usage = openai.Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		}
	}

	return resp
}

func translateMessages(msgs []openai.Message) ([]types.SystemContentBlock, []types.Message, error) {
	var system []types.SystemContentBlock
	var messages []types.Message

	for _, msg := range msgs {
		switch msg.Role {
		case "system", "developer":
			text := msg.ContentString()
			if text != "" {
				system = append(system, &types.SystemContentBlockMemberText{Value: text})
			}

		case "user":
			content, err := translateContentBlocks(msg)
			if err != nil {
				return nil, nil, err
			}
			messages = appendOrCoalesce(messages, types.ConversationRoleUser, content)

		case "assistant":
			var content []types.ContentBlock
			text := msg.ContentString()
			if text != "" {
				content = append(content, &types.ContentBlockMemberText{Value: text})
			}
			// Handle tool calls in assistant messages
			for _, tc := range msg.ToolCalls {
				var inputDoc map[string]interface{}
				if tc.Function.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &inputDoc); err != nil {
						inputDoc = map[string]interface{}{"raw": tc.Function.Arguments}
					}
				}
				content = append(content, &types.ContentBlockMemberToolUse{
					Value: types.ToolUseBlock{
						ToolUseId: aws.String(tc.ID),
						Name:      aws.String(tc.Function.Name),
						Input:     document.NewLazyDocument(inputDoc),
					},
				})
			}
			if len(content) > 0 {
				messages = appendOrCoalesce(messages, types.ConversationRoleAssistant, content)
			}

		case "tool":
			// Tool results become user messages with toolResult content blocks
			var resultContent interface{}
			text := msg.ContentString()
			if text != "" {
				resultContent = text
			}
			contentJSON, _ := json.Marshal(resultContent)
			content := []types.ContentBlock{
				&types.ContentBlockMemberToolResult{
					Value: types.ToolResultBlock{
						ToolUseId: aws.String(msg.ToolCallID),
						Content: []types.ToolResultContentBlock{
							&types.ToolResultContentBlockMemberText{Value: string(contentJSON)},
						},
					},
				},
			}
			messages = appendOrCoalesce(messages, types.ConversationRoleUser, content)

		default:
			return nil, nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	return system, messages, nil
}

func translateContentBlocks(msg openai.Message) ([]types.ContentBlock, error) {
	if len(msg.Content) == 0 {
		return []types.ContentBlock{&types.ContentBlockMemberText{Value: ""}}, nil
	}

	// Try as plain string
	var s string
	if err := json.Unmarshal(msg.Content, &s); err == nil {
		return []types.ContentBlock{&types.ContentBlockMemberText{Value: s}}, nil
	}

	// Try as array of content parts
	var parts []openai.ContentPart
	if err := json.Unmarshal(msg.Content, &parts); err != nil {
		return nil, fmt.Errorf("failed to parse message content: %w", err)
	}

	var blocks []types.ContentBlock
	for _, p := range parts {
		switch p.Type {
		case "text":
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		default:
			// Skip unsupported content types (image_url, etc.) for now
		}
	}

	if len(blocks) == 0 {
		blocks = append(blocks, &types.ContentBlockMemberText{Value: ""})
	}

	return blocks, nil
}

// appendOrCoalesce adds content to messages, merging into the last message if same role.
func appendOrCoalesce(messages []types.Message, role types.ConversationRole, content []types.ContentBlock) []types.Message {
	if len(messages) > 0 && messages[len(messages)-1].Role == role {
		messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, content...)
		return messages
	}
	return append(messages, types.Message{
		Role:    role,
		Content: content,
	})
}

func translateInferenceConfig(req *openai.ChatCompletionRequest, modelID string) *types.InferenceConfiguration {
	cfg := &types.InferenceConfiguration{}
	hasField := false

	if req.MaxTokens != nil {
		cfg.MaxTokens = req.MaxTokens
		hasField = true
	}
	if req.Temperature != nil && !modelRejectsTemperature(modelID) {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
		hasField = true
	}
	if req.TopP != nil {
		p := float32(*req.TopP)
		cfg.TopP = &p
		hasField = true
	}
	if len(req.Stop) > 0 {
		cfg.StopSequences = req.Stop
		hasField = true
	}

	if !hasField {
		return nil
	}
	return cfg
}

func modelRejectsTemperature(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, needle := range modelsWithoutTemperature {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}

func translateToolConfig(req *openai.ChatCompletionRequest) (*types.ToolConfiguration, error) {
	if len(req.Tools) == 0 {
		return nil, nil
	}

	var tools []types.Tool
	for _, t := range req.Tools {
		if t.Type != "function" {
			continue
		}

		spec := types.ToolSpecification{
			Name:        aws.String(t.Function.Name),
			Description: aws.String(t.Function.Description),
		}

		if len(t.Function.Parameters) > 0 {
			var schema map[string]interface{}
			if err := json.Unmarshal(t.Function.Parameters, &schema); err != nil {
				return nil, fmt.Errorf("invalid tool parameters schema: %w", err)
			}
			spec.InputSchema = &types.ToolInputSchemaMemberJson{Value: toDocument(schema)}
		}

		tools = append(tools, &types.ToolMemberToolSpec{Value: spec})
	}

	toolCfg := &types.ToolConfiguration{
		Tools: tools,
	}

	// Handle tool_choice
	if len(req.ToolChoice) > 0 {
		var choiceStr string
		if err := json.Unmarshal(req.ToolChoice, &choiceStr); err == nil {
			switch choiceStr {
			case "auto":
				toolCfg.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
			case "required":
				toolCfg.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
			case "none":
				// No tool use — remove tool config entirely
				return nil, nil
			}
		} else {
			// Try as object: {"type": "function", "function": {"name": "..."}}
			var choiceObj struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(req.ToolChoice, &choiceObj); err == nil && choiceObj.Function.Name != "" {
				toolCfg.ToolChoice = &types.ToolChoiceMemberTool{
					Value: types.SpecificToolChoice{
						Name: aws.String(choiceObj.Function.Name),
					},
				}
			}
		}
	}

	return toolCfg, nil
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

// toDocument converts a Go map to a Bedrock document interface.
func toDocument(v map[string]interface{}) document.Interface {
	if v == nil {
		v = map[string]interface{}{}
	}
	return document.NewLazyDocument(v)
}
