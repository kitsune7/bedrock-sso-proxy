package handler

import (
	"encoding/json"
	"net/http"

	"bedrock-sso-proxy/internal/models"
)

// ModelsHandler handles GET /v1/models.
type ModelsHandler struct {
	registry *models.Registry
}

func NewModelsHandler(registry *models.Registry) *ModelsHandler {
	return &ModelsHandler{registry: registry}
}

func (h *ModelsHandler) Handle(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.registry.ListModels())
}
