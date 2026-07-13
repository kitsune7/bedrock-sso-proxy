package models

import (
	"strings"
	"time"

	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/openai"
)

type Model struct {
	Alias       string
	BedrockID   string
	OwnedBy     string
	CrossRegion bool
}

var defaultModels = []Model{
	{Alias: "claude-fable-5", BedrockID: "anthropic.claude-fable-5", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-opus-4-8", BedrockID: "anthropic.claude-opus-4-8", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-opus-4-7", BedrockID: "anthropic.claude-opus-4-7", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-opus-4-6", BedrockID: "anthropic.claude-opus-4-6-v1", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-sonnet-5", BedrockID: "anthropic.claude-sonnet-5", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-sonnet-4-6", BedrockID: "anthropic.claude-sonnet-4-6", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-sonnet-4-5", BedrockID: "anthropic.claude-sonnet-4-5-20250929-v1:0", OwnedBy: "anthropic", CrossRegion: true},
	{Alias: "claude-haiku-4-5", BedrockID: "anthropic.claude-haiku-4-5-20251001-v1:0", OwnedBy: "anthropic", CrossRegion: true},
}

type Registry struct {
	models      []Model
	aliasMap    map[string]Model
	cfg         *config.Config
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

// ResolveModelID takes a model name from the request and returns the full Bedrock model ID.
func (r *Registry) ResolveModelID(input string) string {
	if input == "" {
		input = r.cfg.DefaultModel
	}

	// Try alias lookup as-is, then fall back to stripping an Ollama-style
	// ":tag" suffix (e.g. "claude-opus-4-7:latest"). Bedrock IDs legitimately
	// contain colons (e.g. "...-v1:0"), so the as-is lookup handles those.
	if m, ok := r.aliasMap[strings.ToLower(input)]; ok {
		return r.applyRegionPrefix(m.BedrockID, m.CrossRegion)
	}
	if i := strings.LastIndex(input, ":"); i > 0 {
		if m, ok := r.aliasMap[strings.ToLower(input[:i])]; ok {
			return r.applyRegionPrefix(m.BedrockID, m.CrossRegion)
		}
	}

	// If it already has a region prefix (e.g., "us.anthropic.claude-..."), pass through
	if hasRegionPrefix(input) {
		return input
	}

	// Unknown model — pass through with optional region prefix
	if r.cfg.CrossRegion {
		return regionPrefix(r.cfg.AWSRegion) + input
	}
	return input
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

func hasRegionPrefix(id string) bool {
	prefixes := []string{"us.", "eu.", "ap.", "global."}
	lower := strings.ToLower(id)
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
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
