// Package codexauth implements OpenAI Codex Agent Identity authentication.
//
// The sandbox sends a harmless placeholder bearer token. iron-proxy resolves
// the real Agent Identity JWT, registers an agent task, and replaces the
// placeholder with the per-request AgentAssertion header that OpenAI expects.
//
// The transform trusts the configured secret source to supply the JWT. It does
// not validate the JWT signature against OpenAI keys, but it does validate the
// stable issuer/audience, expiration, and required private-key claims before
// using the token to sign upstream requests.
//
// Like all header-injecting transforms, this requires MITM mode; sni-only
// mode has no way to rewrite headers.
package codexauth

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/nacl/box"
	"golang.org/x/sync/singleflight"
	"gopkg.in/yaml.v3"

	"github.com/ironsh/iron-proxy/internal/hostmatch"
	"github.com/ironsh/iron-proxy/internal/transform"
	"github.com/ironsh/iron-proxy/internal/transform/secrets"
)

const (
	defaultAuthAPIBaseURL = "https://auth.openai.com/api/accounts"
	defaultHeader         = "Authorization"
	defaultPlaceholder    = "CODEX_ACCESS_TOKEN"
	expectedIssuer        = "https://chatgpt.com/codex-backend/agent-identity"
	expectedAudience      = "codex-app-server"
	registerTimeout       = 30 * time.Second
)

func init() {
	transform.Register("codex_agent_identity", factory)
}

type config struct {
	Identities []identityConfig `yaml:"identities"`
}

type identityConfig struct {
	Source         yaml.Node              `yaml:"source"`
	Rules          []hostmatch.RuleConfig `yaml:"rules"`
	Header         string                 `yaml:"header,omitempty"`
	Placeholder    string                 `yaml:"placeholder,omitempty"`
	AuthAPIBaseURL string                 `yaml:"authapi_base_url,omitempty"`
}

type sourceBuilder func(yaml.Node, *slog.Logger) (secrets.Source, error)

// CodexAuth is the transform.
type CodexAuth struct {
	entries []*identityEntry
}

type identityEntry struct {
	source         secrets.Source
	rules          []hostmatch.Rule
	header         string
	placeholder    string
	authAPIBaseURL string
	httpClient     *http.Client
	logger         *slog.Logger
	now            func() time.Time

	mu     sync.Mutex
	state  *identityState
	flight singleflight.Group
}

type identityState struct {
	fingerprint [32]byte
	claims      agentIdentityClaims
	privateKey  ed25519.PrivateKey
	taskID      string
}

type requestAuth struct {
	authorization string
	accountID     string
	fedramp       bool
}

type agentIdentityClaims struct {
	Iss                     string `json:"iss"`
	Aud                     string `json:"aud"`
	Exp                     int64  `json:"exp"`
	AgentRuntimeID          string `json:"agent_runtime_id"`
	AgentPrivateKey         string `json:"agent_private_key"`
	AccountID               string `json:"account_id"`
	ChatGPTAccountIsFedRAMP bool   `json:"chatgpt_account_is_fedramp"`
}

type registerTaskRequest struct {
	Timestamp string `json:"timestamp"`
	Signature string `json:"signature"`
}

type registerTaskResponse struct {
	TaskID               string `json:"task_id"`
	TaskIDCamel          string `json:"taskId"`
	EncryptedTaskID      string `json:"encrypted_task_id"`
	EncryptedTaskIDCamel string `json:"encryptedTaskId"`
}

type agentAssertionEnvelope struct {
	AgentRuntimeID string `json:"agent_runtime_id"`
	Signature      string `json:"signature"`
	TaskID         string `json:"task_id"`
	Timestamp      string `json:"timestamp"`
}

func factory(cfg yaml.Node, logger *slog.Logger) (transform.Transformer, error) {
	var c config
	if err := cfg.Decode(&c); err != nil {
		return nil, fmt.Errorf("parsing codex_agent_identity config: %w", err)
	}
	return newFromConfig(c, logger, secrets.BuildSource)
}

func newFromConfig(c config, logger *slog.Logger, build sourceBuilder) (*CodexAuth, error) {
	if len(c.Identities) == 0 {
		return nil, fmt.Errorf("codex_agent_identity: at least one entry in \"identities\" is required")
	}

	entries := make([]*identityEntry, 0, len(c.Identities))
	for i, ic := range c.Identities {
		entry, err := buildEntry(ic, logger, build)
		if err != nil {
			return nil, fmt.Errorf("codex_agent_identity: identities[%d]: %w", i, err)
		}
		entries = append(entries, entry)
	}
	return &CodexAuth{entries: entries}, nil
}

func buildEntry(ic identityConfig, logger *slog.Logger, build sourceBuilder) (*identityEntry, error) {
	src, err := build(ic.Source, logger)
	if err != nil {
		return nil, fmt.Errorf("building source: %w", err)
	}

	rules, err := hostmatch.CompileRules(ic.Rules, "codex_agent_identity")
	if err != nil {
		return nil, err
	}
	if len(rules) == 0 {
		return nil, fmt.Errorf("at least one entry in \"rules\" is required")
	}

	header := ic.Header
	if header == "" {
		header = defaultHeader
	}
	placeholder := ic.Placeholder
	if placeholder == "" {
		placeholder = defaultPlaceholder
	}
	baseURL := strings.TrimRight(ic.AuthAPIBaseURL, "/")
	if baseURL == "" {
		baseURL = defaultAuthAPIBaseURL
	}

	return &identityEntry{
		source:         src,
		rules:          rules,
		header:         header,
		placeholder:    placeholder,
		authAPIBaseURL: baseURL,
		httpClient:     &http.Client{Timeout: registerTimeout},
		logger:         logger,
		now:            time.Now,
	}, nil
}

func (c *CodexAuth) Name() string { return "codex_agent_identity" }

func (c *CodexAuth) TransformRequest(ctx context.Context, tctx *transform.TransformContext, req *http.Request) (*transform.TransformResult, error) {
	entry := c.matchEntry(req)
	if entry == nil {
		return &transform.TransformResult{Action: transform.ActionContinue}, nil
	}

	auth, err := entry.requestAuth(ctx)
	if err != nil {
		tctx.Annotate("error", err.Error())
		tctx.Annotate("rejected", "agent_identity_unavailable")
		return &transform.TransformResult{
			Action:   transform.ActionReject,
			Response: errorResponse(req, http.StatusBadGateway, "agent_identity_unavailable"),
		}, nil
	}

	transform.SetHeaderPreservingCase(req.Header, entry.header, auth.authorization)
	transform.SetHeaderPreservingCase(req.Header, "ChatGPT-Account-ID", auth.accountID)
	injected := []string{
		"header:" + http.CanonicalHeaderKey(entry.header),
		"header:ChatGPT-Account-ID",
	}
	if auth.fedramp {
		transform.SetHeaderPreservingCase(req.Header, "X-OpenAI-Fedramp", "true")
		injected = append(injected, "header:X-OpenAI-Fedramp")
	}
	tctx.Annotate("injected", injected)
	tctx.Annotate("auth_scheme", "AgentAssertion")
	return &transform.TransformResult{Action: transform.ActionContinue}, nil
}

func (c *CodexAuth) TransformResponse(context.Context, *transform.TransformContext, *http.Request, *http.Response) (*transform.TransformResult, error) {
	return &transform.TransformResult{Action: transform.ActionContinue}, nil
}

func (c *CodexAuth) matchEntry(req *http.Request) *identityEntry {
	for _, entry := range c.entries {
		if !hostmatch.MatchAnyRule(entry.rules, req) {
			continue
		}
		if strings.Contains(req.Header.Get(entry.header), entry.placeholder) {
			return entry
		}
	}
	return nil
}

func (e *identityEntry) requestAuth(ctx context.Context) (requestAuth, error) {
	state, err := e.loadState(ctx)
	if err != nil {
		return requestAuth{}, err
	}
	timestamp := e.now().UTC().Format(time.RFC3339)
	payload := state.claims.AgentRuntimeID + ":" + state.taskID + ":" + timestamp
	sig := ed25519.Sign(state.privateKey, []byte(payload))
	envelope := agentAssertionEnvelope{
		AgentRuntimeID: state.claims.AgentRuntimeID,
		Signature:      base64.StdEncoding.EncodeToString(sig),
		TaskID:         state.taskID,
		Timestamp:      timestamp,
	}
	raw, err := json.Marshal(envelope)
	if err != nil {
		return requestAuth{}, fmt.Errorf("serializing agent assertion: %w", err)
	}
	return requestAuth{
		authorization: "AgentAssertion " + base64.RawURLEncoding.EncodeToString(raw),
		accountID:     state.claims.AccountID,
		fedramp:       state.claims.ChatGPTAccountIsFedRAMP,
	}, nil
}

func (e *identityEntry) loadState(ctx context.Context) (*identityState, error) {
	token, err := e.source.Get(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving agent identity token: %w", err)
	}
	fingerprint := sha256.Sum256([]byte(token))

	if state, ok := e.cachedState(fingerprint, e.now()); ok {
		return state, nil
	}

	// Deduplicate concurrent first-hit registrations for the same fingerprint
	// so the upstream HTTP call runs without holding e.mu.
	key := string(fingerprint[:])
	v, err, _ := e.flight.Do(key, func() (any, error) {
		if state, ok := e.cachedState(fingerprint, e.now()); ok {
			return state, nil
		}
		claims, privateKey, err := parseAgentIdentityToken(token, e.now())
		if err != nil {
			return nil, err
		}
		taskID, err := e.registerTask(ctx, claims, privateKey)
		if err != nil {
			return nil, err
		}
		state := &identityState{
			fingerprint: fingerprint,
			claims:      claims,
			privateKey:  privateKey,
			taskID:      taskID,
		}
		e.mu.Lock()
		e.state = state
		e.mu.Unlock()
		return state, nil
	})
	if err != nil {
		return nil, err
	}
	return v.(*identityState), nil
}

// cachedState returns the entry's cached state if its fingerprint matches and
// its claims are still valid. A fingerprint match with expired/invalid claims
// drops the cache so the next caller registers fresh.
func (e *identityEntry) cachedState(fingerprint [32]byte, now time.Time) (*identityState, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.state == nil || e.state.fingerprint != fingerprint {
		return nil, false
	}
	if err := validateAgentIdentityClaims(e.state.claims, now); err != nil {
		e.state = nil
		return nil, false
	}
	return e.state, true
}

func (e *identityEntry) registerTask(ctx context.Context, claims agentIdentityClaims, privateKey ed25519.PrivateKey) (string, error) {
	timestamp := e.now().UTC().Format(time.RFC3339)
	payload := claims.AgentRuntimeID + ":" + timestamp
	signature := base64.StdEncoding.EncodeToString(ed25519.Sign(privateKey, []byte(payload)))
	body, err := json.Marshal(registerTaskRequest{Timestamp: timestamp, Signature: signature})
	if err != nil {
		return "", fmt.Errorf("serializing agent task registration: %w", err)
	}

	url := e.authAPIBaseURL + "/v1/agent/" + claims.AgentRuntimeID + "/task/register"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(body)))
	if err != nil {
		return "", fmt.Errorf("building agent task registration request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("registering agent task: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Surface only the status code to the caller (and therefore to audit
		// annotations). The response body is intentionally not read or logged:
		// the upstream — or any intermediary on the path to it — could echo
		// back request material that callers consider sensitive. Status plus
		// content_length is enough to distinguish empty responses from those
		// that carried a payload without exposing the payload itself.
		if e.logger != nil {
			e.logger.Debug("codex agent task registration failed",
				slog.String("status", resp.Status),
				slog.String("agent_runtime_id", claims.AgentRuntimeID),
				slog.Int64("response_content_length", resp.ContentLength),
			)
		}
		return "", fmt.Errorf("registering agent task: status %s", resp.Status)
	}

	var out registerTaskResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decoding agent task registration response: %w", err)
	}
	if out.TaskID != "" {
		return out.TaskID, nil
	}
	if out.TaskIDCamel != "" {
		return out.TaskIDCamel, nil
	}
	encrypted := out.EncryptedTaskID
	if encrypted == "" {
		encrypted = out.EncryptedTaskIDCamel
	}
	if encrypted == "" {
		return "", fmt.Errorf("agent task registration response omitted task id")
	}
	taskID, err := decryptTaskID(privateKey, encrypted)
	if err != nil {
		return "", err
	}
	return taskID, nil
}

func parseAgentIdentityToken(token string, now time.Time) (agentIdentityClaims, ed25519.PrivateKey, error) {
	var claims agentIdentityClaims
	parts := strings.Split(token, ".")
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return claims, nil, fmt.Errorf("invalid agent identity JWT format")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return claims, nil, fmt.Errorf("decoding agent identity JWT payload: %w", err)
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return claims, nil, fmt.Errorf("decoding agent identity JWT claims: %w", err)
	}
	if err := validateAgentIdentityClaims(claims, now); err != nil {
		return claims, nil, err
	}
	privateKey, err := parseEd25519PKCS8PrivateKey(claims.AgentPrivateKey)
	if err != nil {
		return claims, nil, err
	}
	return claims, privateKey, nil
}

func validateAgentIdentityClaims(claims agentIdentityClaims, now time.Time) error {
	if claims.Iss != expectedIssuer {
		return fmt.Errorf("agent identity JWT has invalid issuer")
	}
	if claims.Aud != expectedAudience {
		return fmt.Errorf("agent identity JWT has invalid audience")
	}
	if claims.Exp == 0 {
		return fmt.Errorf("agent identity JWT missing expiration")
	}
	if now.Unix() >= claims.Exp {
		return fmt.Errorf("agent identity JWT expired")
	}
	if claims.AgentRuntimeID == "" || claims.AgentPrivateKey == "" || claims.AccountID == "" {
		return fmt.Errorf("agent identity JWT missing required claims")
	}
	return nil
}

func parseEd25519PKCS8PrivateKey(privateKeyPKCS8Base64 string) (ed25519.PrivateKey, error) {
	der, err := base64.StdEncoding.DecodeString(privateKeyPKCS8Base64)
	if err != nil {
		return nil, fmt.Errorf("decoding agent private key: %w", err)
	}
	key, err := x509.ParsePKCS8PrivateKey(der)
	if err != nil {
		return nil, fmt.Errorf("parsing agent private key: %w", err)
	}
	privateKey, ok := key.(ed25519.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("agent private key is %T, want ed25519.PrivateKey", key)
	}
	return privateKey, nil
}

func decryptTaskID(privateKey ed25519.PrivateKey, encryptedTaskID string) (string, error) {
	ciphertext, err := base64.StdEncoding.DecodeString(encryptedTaskID)
	if err != nil {
		return "", fmt.Errorf("decoding encrypted task id: %w", err)
	}

	var curvePrivate [32]byte
	digest := sha512.Sum512(privateKey.Seed())
	copy(curvePrivate[:], digest[:32])
	curvePrivate[0] &= 248
	curvePrivate[31] &= 127
	curvePrivate[31] |= 64

	curvePublicBytes, err := curve25519.X25519(curvePrivate[:], curve25519.Basepoint)
	if err != nil {
		return "", fmt.Errorf("deriving task id decrypt public key: %w", err)
	}
	var curvePublic [32]byte
	copy(curvePublic[:], curvePublicBytes)

	plaintext, ok := box.OpenAnonymous(nil, ciphertext, &curvePublic, &curvePrivate)
	if !ok {
		return "", fmt.Errorf("decrypting encrypted task id")
	}
	return string(plaintext), nil
}

func errorResponse(req *http.Request, status int, reason string) *http.Response {
	body := []byte(`{"error":"codex_agent_identity","reason":"` + reason + `"}`)
	return &http.Response{
		StatusCode:    status,
		Status:        strconv.Itoa(status) + " " + http.StatusText(status),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        http.Header{"Content-Type": {"application/json"}},
		Body:          transform.NewBufferedBodyFromBytes(body),
		ContentLength: int64(len(body)),
		Request:       req,
	}
}
