package main

import (
	"fmt"
	"log"
	"net/http"

	"bedrock-sso-proxy/internal/auth"
	"bedrock-sso-proxy/internal/config"
	"bedrock-sso-proxy/internal/handler"
	"bedrock-sso-proxy/internal/models"
)

func main() {
	cfg := config.Parse()

	log.Printf("Starting Bedrock SSO Proxy (profile=%s, region=%s, default-model=%s)",
		cfg.AWSProfile, cfg.AWSRegion, cfg.DefaultModel)

	authMgr, err := auth.NewManager(cfg.AWSProfile, cfg.AWSRegion)
	if err != nil {
		log.Fatalf("Failed to initialize AWS auth: %v", err)
	}
	log.Println("AWS credentials loaded successfully")

	registry := models.NewRegistry(cfg)
	chatHandler := handler.NewChatHandler(authMgr, registry, cfg)
	modelsHandler := handler.NewModelsHandler(registry)
	ollamaChatHandler := handler.NewOllamaChatHandler(authMgr, registry, cfg)
	ollamaTagsHandler := handler.NewOllamaTagsHandler(registry)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", chatHandler.Handle)
	mux.HandleFunc("GET /v1/models", modelsHandler.Handle)
	mux.HandleFunc("POST /api/chat", ollamaChatHandler.Handle)
	mux.HandleFunc("GET /api/tags", ollamaTagsHandler.Handle)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	wrapped := handler.Chain(mux, handler.Logger, handler.Recovery, handler.CORS)

	addr := fmt.Sprintf(":%d", cfg.Port)
	log.Printf("Listening on http://localhost%s", addr)
	log.Printf("Default model: %s -> %s", cfg.DefaultModel, registry.ResolveModelID(cfg.DefaultModel))
	log.Fatal(http.ListenAndServe(addr, wrapped))
}
