package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

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

func TestTranslateContentBlocks_Images(t *testing.T) {
	// A 1x1 PNG. Converse takes raw bytes or an S3 location — never a URL — so
	// the translation has to decode the payload rather than forward it.
	const pngB64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="

	raw := func(v any) json.RawMessage {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	t.Run("data URL becomes an image block with bytes", func(t *testing.T) {
		msg := openai.Message{Role: "user", Content: raw([]map[string]any{
			{"type": "text", "text": "what is this"},
			{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + pngB64}},
		})}

		blocks, err := translateContentBlocks(context.Background(), msg)
		if err != nil {
			t.Fatalf("translateContentBlocks: %v", err)
		}
		if len(blocks) != 2 {
			t.Fatalf("len(blocks) = %d, want 2", len(blocks))
		}
		if _, ok := blocks[0].(*types.ContentBlockMemberText); !ok {
			t.Errorf("blocks[0] = %T, want a text block", blocks[0])
		}
		img, ok := blocks[1].(*types.ContentBlockMemberImage)
		if !ok {
			t.Fatalf("blocks[1] = %T, want an image block", blocks[1])
		}
		if img.Value.Format != types.ImageFormatPng {
			t.Errorf("Format = %q, want png", img.Value.Format)
		}
		src, ok := img.Value.Source.(*types.ImageSourceMemberBytes)
		if !ok {
			t.Fatalf("Source = %T, want bytes", img.Value.Source)
		}
		want, err := base64.StdEncoding.DecodeString(pngB64)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(src.Value, want) {
			t.Error("image bytes do not match the decoded payload")
		}
	})

	t.Run("s3 URI becomes an S3 location", func(t *testing.T) {
		msg := openai.Message{Role: "user", Content: raw([]map[string]any{
			{"type": "image_url", "image_url": map[string]string{"url": "s3://bucket/pic.jpg"}},
		})}

		blocks, err := translateContentBlocks(context.Background(), msg)
		if err != nil {
			t.Fatalf("translateContentBlocks: %v", err)
		}
		img := blocks[0].(*types.ContentBlockMemberImage)
		if img.Value.Format != types.ImageFormatJpeg {
			t.Errorf("Format = %q, want jpeg from the extension", img.Value.Format)
		}
		src, ok := img.Value.Source.(*types.ImageSourceMemberS3Location)
		if !ok {
			t.Fatalf("Source = %T, want an S3 location", img.Value.Source)
		}
		if aws.ToString(src.Value.Uri) != "s3://bucket/pic.jpg" {
			t.Errorf("Uri = %q, want the URI unchanged", aws.ToString(src.Value.Uri))
		}
	})

	// Images used to be dropped here, which sent a vision request to the model as
	// text-only and produced a confidently wrong answer with no error.
	t.Run("unusable image is an error, not a drop", func(t *testing.T) {
		msg := openai.Message{Role: "user", Content: raw([]map[string]any{
			{"type": "image_url", "image_url": map[string]string{"url": "ftp://host/pic.png"}},
		})}
		if _, err := translateContentBlocks(context.Background(), msg); err == nil {
			t.Error("expected an error for an unsupported image scheme")
		}
	})

	t.Run("plain string content still works", func(t *testing.T) {
		blocks, err := translateContentBlocks(context.Background(), openai.Message{Role: "user", Content: raw("hi")})
		if err != nil {
			t.Fatalf("translateContentBlocks: %v", err)
		}
		text, ok := blocks[0].(*types.ContentBlockMemberText)
		if !ok || text.Value != "hi" {
			t.Errorf("blocks[0] = %+v, want a text block with hi", blocks[0])
		}
	})
}
