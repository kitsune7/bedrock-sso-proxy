package bedrock

import (
	"testing"

	"bedrock-sso-proxy/internal/openai"
)

func TestTranslateInferenceConfig_DropsTemperatureForOpus47(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)
	req := &openai.ChatCompletionRequest{
		Temperature: &temp,
		TopP:        &topP,
	}

	cases := []struct {
		name       string
		modelID    string
		wantTempNil bool
	}{
		{"opus 4.7 alias", "claude-opus-4-7", true},
		{"opus 4.7 bedrock id", "anthropic.claude-opus-4-7", true},
		{"opus 4.7 region-prefixed", "us.anthropic.claude-opus-4-7", true},
		{"opus 4.6 keeps temperature", "anthropic.claude-opus-4-6-v1", false},
		{"sonnet 4.6 keeps temperature", "us.anthropic.claude-sonnet-4-6", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := translateInferenceConfig(req, tc.modelID)
			if cfg == nil {
				t.Fatal("expected non-nil config")
			}
			gotNil := cfg.Temperature == nil
			if gotNil != tc.wantTempNil {
				t.Errorf("Temperature nil = %v, want %v (modelID=%q)", gotNil, tc.wantTempNil, tc.modelID)
			}
			if cfg.TopP == nil {
				t.Error("TopP should pass through regardless of model")
			}
		})
	}
}
