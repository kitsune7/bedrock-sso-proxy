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

func TestResolveModelID_Opus5(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-5",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	cases := []struct {
		in   string
		want string
	}{
		{"", "us.anthropic.claude-opus-5"},
		{"claude-opus-5", "us.anthropic.claude-opus-5"},
		{"claude-opus-5:latest", "us.anthropic.claude-opus-5"},
		{"anthropic.claude-opus-5", "us.anthropic.claude-opus-5"},
		{"us.anthropic.claude-opus-5", "us.anthropic.claude-opus-5"},
	}
	for _, tc := range cases {
		if got := r.ResolveModelID(tc.in); got != tc.want {
			t.Errorf("ResolveModelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestResolve_BackendAndRegionPrefix(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-5",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	cases := []struct {
		in          string
		wantID      string
		wantBackend Backend
	}{
		// Claude: cross-region prefix applies, Converse backend by default.
		{"claude-opus-5", "us.anthropic.claude-opus-5", BackendConverse},
		// gpt-oss has no us./global. profile — the prefix must NOT be added, or
		// Bedrock 400s on a nonexistent inference profile.
		{"gpt-oss-120b", "openai.gpt-oss-120b-1:0", BackendConverse},
		{"gpt-oss-120b:latest", "openai.gpt-oss-120b-1:0", BackendConverse},
		{"openai.gpt-oss-20b-1:0", "openai.gpt-oss-20b-1:0", BackendConverse},
		// GPT-5 is Responses-API only, and likewise has no inference profile.
		{"gpt-5.6-sol", "openai.gpt-5.6-sol", BackendMantleResponses},
		{"gpt-5.6-sol:latest", "openai.gpt-5.6-sol", BackendMantleResponses},
		{"openai.gpt-5.6-sol", "openai.gpt-5.6-sol", BackendMantleResponses},
		{"gpt-5.6-terra", "openai.gpt-5.6-terra", BackendMantleResponses},
		{"gpt-5.6-luna", "openai.gpt-5.6-luna", BackendMantleResponses},
		{"gpt-5.5", "openai.gpt-5.5", BackendMantleResponses},
		{"gpt-5.5:latest", "openai.gpt-5.5", BackendMantleResponses},
		{"openai.gpt-5.5", "openai.gpt-5.5", BackendMantleResponses},
		{"gpt-5.4", "openai.gpt-5.4", BackendMantleResponses},
		// Unknown models fall through to Converse with a prefix.
		{"some.unknown-model", "us.some.unknown-model", BackendConverse},
	}
	for _, tc := range cases {
		got := r.Resolve(tc.in)
		if got.ID != tc.wantID {
			t.Errorf("Resolve(%q).ID = %q, want %q", tc.in, got.ID, tc.wantID)
		}
		if got.Backend != tc.wantBackend {
			t.Errorf("Resolve(%q).Backend = %q, want %q", tc.in, got.Backend, tc.wantBackend)
		}
	}
}

func TestResolve_Traits(t *testing.T) {
	cfg := &config.Config{
		DefaultModel: "claude-opus-5",
		AWSRegion:    "us-east-1",
		CrossRegion:  true,
	}
	r := NewRegistry(cfg)

	cases := []struct {
		in                    string
		wantNoSampling        bool
		wantExclusiveSampling bool
		wantReasoning         bool
	}{
		{"claude-fable-5", true, false, true},
		{"claude-opus-5", true, false, true},
		{"claude-opus-4-8", true, false, true},
		{"claude-opus-4-7", true, false, true},
		{"claude-sonnet-5", true, false, true},
		{"gpt-5.6-sol", true, false, true},
		{"gpt-5.6-terra", true, false, true},
		{"gpt-5.6-luna", true, false, true},
		{"gpt-5.5", true, false, true},
		// gpt-oss reasons, and unlike Anthropic accepts temperature and top_p
		// in the same request.
		{"gpt-oss-120b", false, false, true},
		// Sampling-capable Anthropic models take one param or the other.
		{"claude-opus-4-6", false, true, false},
		{"claude-sonnet-4-6", false, true, false},
		{"claude-haiku-4-5", false, true, false},
		// A fully-qualified inference profile ID must resolve to its registry
		// entry, traits included — otherwise a client sending the prefixed form
		// would get temperature forwarded to a model that 400s on it.
		{"us.anthropic.claude-opus-5", true, false, true},
		{"anthropic.claude-opus-5", true, false, true},
	}
	for _, tc := range cases {
		got := r.Resolve(tc.in)
		if got.NoSampling != tc.wantNoSampling {
			t.Errorf("Resolve(%q).NoSampling = %v, want %v", tc.in, got.NoSampling, tc.wantNoSampling)
		}
		if got.ExclusiveSampling != tc.wantExclusiveSampling {
			t.Errorf("Resolve(%q).ExclusiveSampling = %v, want %v", tc.in, got.ExclusiveSampling, tc.wantExclusiveSampling)
		}
		if got.Reasoning != tc.wantReasoning {
			t.Errorf("Resolve(%q).Reasoning = %v, want %v", tc.in, got.Reasoning, tc.wantReasoning)
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
