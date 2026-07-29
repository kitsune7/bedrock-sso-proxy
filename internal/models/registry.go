package models

import (
	"strings"
	"time"

	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/openai"
)

// Backend names the Bedrock API surface a model is reachable through. Models on
// Bedrock are not all served by the same API, so the registry records which one
// to use and callers dispatch on it rather than assuming Converse.
type Backend string

const (
	// BackendConverse is the Converse API on the bedrock-runtime endpoint.
	BackendConverse Backend = "converse"
	// BackendMantleResponses is the OpenAI Responses API on the bedrock-mantle
	// endpoint, at the non-standard /openai/v1 path.
	BackendMantleResponses Backend = "mantle-responses"
)

// ReasoningMinOutputTokens is the floor the translation layer applies to
// max_tokens for reasoning models. They spend part of the output budget on
// internal reasoning that never reaches the client, so a small max_tokens
// returns an empty completion with finish_reason "length". Anything under this
// is almost certainly a client default rather than a deliberate choice.
const ReasoningMinOutputTokens = 4096

// AdjustMaxTokens raises a reasoning model's output budget to at least
// ReasoningMinOutputTokens. A nil value is left nil so the model's own default
// applies.
func (r Resolved) AdjustMaxTokens(requested *int32) *int32 {
	if requested == nil || !r.Reasoning || *requested >= ReasoningMinOutputTokens {
		return requested
	}
	floor := int32(ReasoningMinOutputTokens)
	return &floor
}

type Model struct {
	Alias     string
	BedrockID string
	OwnedBy   string
	// CrossRegion enables the us./eu./global. inference-profile prefix. Models
	// without published geo or global profiles must leave this false — Bedrock
	// rejects a prefixed ID that has no matching profile.
	CrossRegion bool
	// Backend defaults to BackendConverse when empty.
	Backend Backend
	// NoSampling marks models that reject the sampling parameters
	// (temperature, top_p) with a 400. The translation layer drops them.
	NoSampling bool
	// ExclusiveSampling marks models that accept temperature or top_p but
	// reject both in one request. Anthropic models behave this way; OpenAI's
	// do not. The translation layer keeps temperature and drops top_p.
	ExclusiveSampling bool
	// Reasoning marks models that emit internal reasoning tokens billed
	// against the output budget. See ReasoningMinOutputTokens.
	Reasoning bool
}

var defaultModels = []Model{
	{Alias: "claude-fable-5", BedrockID: "anthropic.claude-fable-5", OwnedBy: "anthropic", CrossRegion: true, NoSampling: true, Reasoning: true},
	{Alias: "claude-opus-5", BedrockID: "anthropic.claude-opus-5", OwnedBy: "anthropic", CrossRegion: true, NoSampling: true, Reasoning: true},
	{Alias: "claude-opus-4-8", BedrockID: "anthropic.claude-opus-4-8", OwnedBy: "anthropic", CrossRegion: true, NoSampling: true, Reasoning: true},
	{Alias: "claude-opus-4-7", BedrockID: "anthropic.claude-opus-4-7", OwnedBy: "anthropic", CrossRegion: true, NoSampling: true, Reasoning: true},
	{Alias: "claude-opus-4-6", BedrockID: "anthropic.claude-opus-4-6-v1", OwnedBy: "anthropic", CrossRegion: true, ExclusiveSampling: true},
	{Alias: "claude-sonnet-5", BedrockID: "anthropic.claude-sonnet-5", OwnedBy: "anthropic", CrossRegion: true, NoSampling: true, Reasoning: true},
	{Alias: "claude-sonnet-4-6", BedrockID: "anthropic.claude-sonnet-4-6", OwnedBy: "anthropic", CrossRegion: true, ExclusiveSampling: true},
	{Alias: "claude-sonnet-4-5", BedrockID: "anthropic.claude-sonnet-4-5-20250929-v1:0", OwnedBy: "anthropic", CrossRegion: true, ExclusiveSampling: true},
	{Alias: "claude-haiku-4-5", BedrockID: "anthropic.claude-haiku-4-5-20251001-v1:0", OwnedBy: "anthropic", CrossRegion: true, ExclusiveSampling: true},

	// OpenAI. gpt-oss speaks Converse on bedrock-runtime, so it rides the same
	// path as Claude. Its only inference profile is us-gov., which this proxy
	// does not target, hence CrossRegion: false.
	{Alias: "gpt-oss-120b", BedrockID: "openai.gpt-oss-120b-1:0", OwnedBy: "openai", Reasoning: true},
	{Alias: "gpt-oss-20b", BedrockID: "openai.gpt-oss-20b-1:0", OwnedBy: "openai", Reasoning: true},

	// The GPT-5 family is reachable only through the OpenAI Responses API on
	// the bedrock-mantle endpoint — no Converse, no Invoke, no bedrock-runtime.
	// No geo or global inference profiles exist, and they are in-region only:
	// us-east-1 and us-east-2 for all of them, plus us-west-2 for 5.6 terra and
	// luna.
	// Aliases keep the vendor's own spelling, so clients configured for OpenAI
	// can point at the proxy unchanged.
	{Alias: "gpt-5.6-sol", BedrockID: "openai.gpt-5.6-sol", OwnedBy: "openai", Backend: BackendMantleResponses, NoSampling: true, Reasoning: true},
	{Alias: "gpt-5.6-terra", BedrockID: "openai.gpt-5.6-terra", OwnedBy: "openai", Backend: BackendMantleResponses, NoSampling: true, Reasoning: true},
	{Alias: "gpt-5.6-luna", BedrockID: "openai.gpt-5.6-luna", OwnedBy: "openai", Backend: BackendMantleResponses, NoSampling: true, Reasoning: true},
	{Alias: "gpt-5.5", BedrockID: "openai.gpt-5.5", OwnedBy: "openai", Backend: BackendMantleResponses, NoSampling: true, Reasoning: true},
	{Alias: "gpt-5.4", BedrockID: "openai.gpt-5.4", OwnedBy: "openai", Backend: BackendMantleResponses, NoSampling: true, Reasoning: true},
}

type Registry struct {
	models   []Model
	aliasMap map[string]Model
	cfg      *config.Config
}

func NewRegistry(cfg *config.Config) *Registry {
	r := &Registry{
		models:   defaultModels,
		aliasMap: make(map[string]Model),
		cfg:      cfg,
	}
	for _, m := range r.models {
		r.aliasMap[strings.ToLower(m.Alias)] = m
		r.aliasMap[strings.ToLower(m.BedrockID)] = m
	}
	return r
}

// Resolved is the outcome of looking up a request's model name: the ID to send
// to Bedrock, which API surface can serve it, and the quirks the translation
// layer has to accommodate.
type Resolved struct {
	ID      string
	Backend Backend
	// NoSampling, ExclusiveSampling, and Reasoning mirror the registry entry's
	// traits.
	NoSampling        bool
	ExclusiveSampling bool
	Reasoning         bool
}

// Resolve takes a model name from the request and returns the full Bedrock model
// ID along with the backend that serves it.
func (r *Registry) Resolve(input string) Resolved {
	if input == "" {
		input = r.cfg.DefaultModel
	}

	// Try alias lookup as-is, then fall back to stripping an Ollama-style
	// ":tag" suffix (e.g. "claude-opus-4-7:latest"). Bedrock IDs legitimately
	// contain colons (e.g. "...-v1:0"), so the as-is lookup handles those.
	// Both forms are tried with any region prefix removed as well, so a client
	// sending a fully-qualified inference profile ID still matches its registry
	// entry and picks up the entry's backend and traits.
	for _, candidate := range []string{input, trimRegionPrefix(input)} {
		if m, ok := r.aliasMap[strings.ToLower(candidate)]; ok {
			return r.resolved(m)
		}
		if i := strings.LastIndex(candidate, ":"); i > 0 {
			if m, ok := r.aliasMap[strings.ToLower(candidate[:i])]; ok {
				return r.resolved(m)
			}
		}
	}

	// Unknown model — pass through, adding a region prefix unless it already has
	// one. Unknown IDs are assumed to speak Converse; anything else needs a
	// registry entry naming its backend.
	id := input
	if !hasRegionPrefix(id) && r.cfg.CrossRegion {
		id = regionPrefix(r.cfg.AWSRegion) + id
	}
	return Resolved{ID: id, Backend: BackendConverse}
}

// ResolveModelID returns just the Bedrock model ID. Kept for callers that only
// need the ID (logging, the Converse path).
func (r *Registry) ResolveModelID(input string) string {
	return r.Resolve(input).ID
}

func (r *Registry) resolved(m Model) Resolved {
	backend := m.Backend
	if backend == "" {
		backend = BackendConverse
	}
	return Resolved{
		ID:                r.applyRegionPrefix(m.BedrockID, m.CrossRegion),
		Backend:           backend,
		NoSampling:        m.NoSampling,
		ExclusiveSampling: m.ExclusiveSampling,
		Reasoning:         m.Reasoning,
	}
}

// ListModels returns the model list for the /v1/models endpoint.
func (r *Registry) ListModels() openai.ModelList {
	created := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC).Unix()
	data := make([]openai.ModelInfo, 0, len(r.models))
	for _, m := range r.models {
		data = append(data, openai.ModelInfo{
			ID:      m.Alias,
			Object:  "model",
			Created: created,
			OwnedBy: m.OwnedBy,
		})
	}
	return openai.ModelList{
		Object: "list",
		Data:   data,
	}
}

var regionPrefixes = []string{"us.", "eu.", "ap.", "global."}

func hasRegionPrefix(id string) bool {
	lower := strings.ToLower(id)
	for _, p := range regionPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
}

// trimRegionPrefix removes a leading inference-profile prefix, returning the
// bare Bedrock model ID. Returns id unchanged when there is no prefix.
func trimRegionPrefix(id string) string {
	lower := strings.ToLower(id)
	for _, p := range regionPrefixes {
		if strings.HasPrefix(lower, p) {
			return id[len(p):]
		}
	}
	return id
}

func regionPrefix(region string) string {
	parts := strings.SplitN(region, "-", 2)
	if len(parts) == 0 {
		return "us."
	}
	return parts[0] + "."
}

func (r *Registry) applyRegionPrefix(bedrockID string, crossRegion bool) string {
	if !r.cfg.CrossRegion || !crossRegion {
		return bedrockID
	}
	if hasRegionPrefix(bedrockID) {
		return bedrockID
	}
	return regionPrefix(r.cfg.AWSRegion) + bedrockID
}
