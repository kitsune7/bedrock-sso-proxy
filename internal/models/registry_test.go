package models

import (
	"testing"

	"bedrock-sso-proxy/internal/config"
)

func TestResolveModelID_DefaultIsOpus47WithRegionPrefix(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-4-7",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	got := r.ResolveModelID("")
	want := "us.anthropic.claude-opus-4-7"
	if got != want {
		t.Errorf("empty input: got %q, want %q", got, want)
	}
}

func TestResolveModelID_Opus47AliasAndFullID(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-4-7",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	cases := []struct {
		in   string
		want string
	}{
		{"claude-opus-4-7", "us.anthropic.claude-opus-4-7"},
		{"anthropic.claude-opus-4-7", "us.anthropic.claude-opus-4-7"},
		{"us.anthropic.claude-opus-4-7", "us.anthropic.claude-opus-4-7"},
	}
	for _, tc := range cases {
		if got := r.ResolveModelID(tc.in); got != tc.want {
			t.Errorf("ResolveModelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestResolveModelID_OllamaTagSuffix(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-4-7",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	// Ollama clients send "model:tag" (e.g. ":latest") — the tag must be
	// stripped before alias lookup so Bedrock sees a valid profile ID.
	cases := []struct {
		in   string
		want string
	}{
		{"claude-opus-4-7:latest", "us.anthropic.claude-opus-4-7"},
		{"claude-sonnet-4-6:latest", "us.anthropic.claude-sonnet-4-6"},
		{"claude-haiku-4-5:custom-tag", "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
		// Bedrock IDs with embedded colons (":0") must still resolve.
		{"anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
	}
	for _, tc := range cases {
		if got := r.ResolveModelID(tc.in); got != tc.want {
			t.Errorf("ResolveModelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestResolveModelID_CrossRegionDisabled(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-4-7",
		AWSRegion:    "us-east-1",
		CrossRegion:  false,
	}
	r := NewRegistry(cfg)

	got := r.ResolveModelID("claude-opus-4-7")
	want := "anthropic.claude-opus-4-7"
	if got != want {
		t.Errorf("CrossRegion=false: got %q, want %q", got, want)
	}
}
