package bedrock

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"

	"bedrock-sso-proxy/internal/models"
	"bedrock-sso-proxy/internal/openai"
)

func TestTranslateInferenceConfig_DropsSamplingParams(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)
	maxTokens := int32(100)
	req := &openai.ChatCompletionRequest{
		Temperature: &temp,
		TopP:        &topP,
		MaxTokens:   &maxTokens,
	}

	cases := []struct {
		name        string
		resolved    models.Resolved
		wantDropped bool
	}{
		{"no-sampling model", models.Resolved{ID: "us.anthropic.claude-opus-5", NoSampling: true}, true},
		{"sampling model", models.Resolved{ID: "us.anthropic.claude-sonnet-4-6"}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := translateInferenceConfig(req, tc.resolved)
			if cfg == nil {
				t.Fatal("expected non-nil config")
			}
			if got := cfg.Temperature == nil; got != tc.wantDropped {
				t.Errorf("Temperature dropped = %v, want %v", got, tc.wantDropped)
			}
			if got := cfg.TopP == nil; got != tc.wantDropped {
				t.Errorf("TopP dropped = %v, want %v", got, tc.wantDropped)
			}
		})
	}
}

// Anthropic models 400 when temperature and top_p are both set. OpenAI clients
// routinely send both, so top_p is dropped rather than failing the request.
func TestTranslateInferenceConfig_ExclusiveSampling(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)

	cases := []struct {
		name        string
		temp, topP  *float32
		resolved    models.Resolved
		wantTempSet bool
		wantTopPSet bool
	}{
		{"both set drops top_p", &temp, &topP, models.Resolved{ExclusiveSampling: true}, true, false},
		{"top_p alone survives", nil, &topP, models.Resolved{ExclusiveSampling: true}, false, true},
		{"temperature alone survives", &temp, nil, models.Resolved{ExclusiveSampling: true}, true, false},
		// Models without the restriction (gpt-oss) accept both.
		{"non-exclusive keeps both", &temp, &topP, models.Resolved{}, true, true},
		{"no-sampling drops both", &temp, &topP, models.Resolved{NoSampling: true}, false, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &openai.ChatCompletionRequest{
				Temperature: tc.temp,
				TopP:        tc.topP,
				MaxTokens:   aws.Int32(100),
			}
			cfg := translateInferenceConfig(req, tc.resolved)
			if cfg == nil {
				t.Fatal("expected non-nil config")
			}
			if got := cfg.Temperature != nil; got != tc.wantTempSet {
				t.Errorf("Temperature set = %v, want %v", got, tc.wantTempSet)
			}
			if got := cfg.TopP != nil; got != tc.wantTopPSet {
				t.Errorf("TopP set = %v, want %v", got, tc.wantTopPSet)
			}
		})
	}
}

func TestTranslateInferenceConfig_ReasoningMaxTokensFloor(t *testing.T) {
	cases := []struct {
		name      string
		requested *int32
		reasoning bool
		want      *int32
	}{
		// A reasoning model spends part of max_tokens on tokens the client never
		// sees, so a small budget yields an empty completion.
		{"raises small budget", aws.Int32(30), true, aws.Int32(models.ReasoningMinOutputTokens)},
		{"leaves large budget", aws.Int32(100000), true, aws.Int32(100000)},
		{"leaves exact floor", aws.Int32(models.ReasoningMinOutputTokens), true, aws.Int32(models.ReasoningMinOutputTokens)},
		{"non-reasoning untouched", aws.Int32(30), false, aws.Int32(30)},
		// nil must stay nil so the model's own default applies rather than the
		// proxy silently capping output.
		{"nil stays nil", nil, true, nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &openai.ChatCompletionRequest{MaxTokens: tc.requested}
			cfg := translateInferenceConfig(req, models.Resolved{Reasoning: tc.reasoning})
			if tc.want == nil {
				if cfg != nil && cfg.MaxTokens != nil {
					t.Fatalf("MaxTokens = %d, want nil", *cfg.MaxTokens)
				}
				return
			}
			if cfg == nil || cfg.MaxTokens == nil {
				t.Fatal("expected MaxTokens to be set")
			}
			if *cfg.MaxTokens != *tc.want {
				t.Errorf("MaxTokens = %d, want %d", *cfg.MaxTokens, *tc.want)
			}
		})
	}
}
