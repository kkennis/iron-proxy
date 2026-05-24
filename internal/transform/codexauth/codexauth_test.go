package codexauth

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha512"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/nacl/box"
	"gopkg.in/yaml.v3"

	"github.com/ironsh/iron-proxy/internal/transform"
	"github.com/ironsh/iron-proxy/internal/transform/secrets"
)

var fixedNow = time.Date(2026, 5, 22, 12, 34, 56, 0, time.UTC)

type staticSource string

func (s staticSource) Name() string { return "static" }

func (s staticSource) Get(context.Context) (string, error) { return string(s), nil }

func staticBuilder(src secrets.Source) sourceBuilder {
	return func(yaml.Node, *slog.Logger) (secrets.Source, error) { return src, nil }
}

func TestTransformRequestInjectsAgentAssertion(t *testing.T) {
	publicKey, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)

	var registerCalls int32
	authAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&registerCalls, 1)
		require.Equal(t, "/v1/agent/runtime-123/task/register", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var body registerTaskRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		sig, err := base64.StdEncoding.DecodeString(body.Signature)
		require.NoError(t, err)
		require.True(t, ed25519.Verify(publicKey, []byte("runtime-123:"+body.Timestamp), sig))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"task_id":"task-456"}`))
	}))
	defer authAPI.Close()

	c := buildTestTransform(t, token, authAPI.URL)
	entry := c.entries[0]
	entry.now = func() time.Time {
		return fixedNow
	}

	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionContinue, res.Action)
	require.Equal(t, int32(1), atomic.LoadInt32(&registerCalls))
	require.Equal(t, []string{"account-abc"}, transform.HeaderValuesByExactName(req.Header, "ChatGPT-Account-ID"))
	require.Empty(t, transform.HeaderValuesByExactName(req.Header, "X-OpenAI-Fedramp"))

	auth := req.Header.Get("Authorization")
	require.True(t, strings.HasPrefix(auth, "AgentAssertion "))
	assertionJSON, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(auth, "AgentAssertion "))
	require.NoError(t, err)
	var assertion agentAssertionEnvelope
	require.NoError(t, json.Unmarshal(assertionJSON, &assertion))
	require.Equal(t, "runtime-123", assertion.AgentRuntimeID)
	require.Equal(t, "task-456", assertion.TaskID)
	require.Equal(t, "2026-05-22T12:34:56Z", assertion.Timestamp)
	sig, err := base64.StdEncoding.DecodeString(assertion.Signature)
	require.NoError(t, err)
	require.True(t, ed25519.Verify(publicKey, []byte("runtime-123:task-456:2026-05-22T12:34:56Z"), sig))
}

func TestTransformRequestSkipsNonMatchingHost(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)
	c := buildTestTransform(t, token, "https://auth.example.test/api/accounts")

	req := httptest.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionContinue, res.Action)
	require.Equal(t, "Bearer CODEX_ACCESS_TOKEN", req.Header.Get("Authorization"))
	require.Empty(t, req.Header.Get("ChatGPT-Account-ID"))
}

func TestTransformRequestSkipsNonMatchingPath(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)
	c := buildTestTransform(t, token, "https://auth.example.test/api/accounts")

	req := httptest.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/other/responses", nil)
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionContinue, res.Action)
	require.Equal(t, "Bearer CODEX_ACCESS_TOKEN", req.Header.Get("Authorization"))
	require.Empty(t, req.Header.Get("ChatGPT-Account-ID"))
}

func TestTransformRequestSkipsNonPlaceholderAuthorization(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)
	c := buildTestTransform(t, token, "https://auth.example.test/api/accounts")

	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer OPENAI_API_KEY")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionContinue, res.Action)
	require.Equal(t, "Bearer OPENAI_API_KEY", req.Header.Get("Authorization"))
	require.Empty(t, req.Header.Get("ChatGPT-Account-ID"))
}

func TestTransformRequestCachesRegisteredTask(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)

	var registerCalls int32
	authAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&registerCalls, 1)
		_, _ = w.Write([]byte(`{"task_id":"task-456"}`))
	}))
	defer authAPI.Close()

	c := buildTestTransform(t, token, authAPI.URL)
	entry := c.entries[0]
	entry.now = func() time.Time {
		return fixedNow
	}

	for range 2 {
		req := newMatchingRequest()
		req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
		_, err := c.TransformRequest(context.Background(), &transform.TransformContext{Mode: transform.ModeMITM}, req)
		require.NoError(t, err)
		require.True(t, strings.HasPrefix(req.Header.Get("Authorization"), "AgentAssertion "))
	}
	require.Equal(t, int32(1), atomic.LoadInt32(&registerCalls))
}

func TestTransformRequestAddsFedRAMPHeader(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", true)

	authAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"task_id":"task-456"}`))
	}))
	defer authAPI.Close()

	c := buildTestTransform(t, token, authAPI.URL)
	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	_, err := c.TransformRequest(context.Background(), &transform.TransformContext{Mode: transform.ModeMITM}, req)
	require.NoError(t, err)
	require.Equal(t, []string{"true"}, transform.HeaderValuesByExactName(req.Header, "X-OpenAI-Fedramp"))
}

func TestTransformRequestAcceptsEncryptedTaskID(t *testing.T) {
	publicKey, privateKey, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)

	authAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encryptedTaskID := encryptTaskID(t, privateKey, "task-789")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"encrypted_task_id":"` + encryptedTaskID + `"}`))
	}))
	defer authAPI.Close()

	c := buildTestTransform(t, token, authAPI.URL)
	entry := c.entries[0]
	entry.now = func() time.Time {
		return fixedNow
	}

	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	_, err := c.TransformRequest(context.Background(), &transform.TransformContext{Mode: transform.ModeMITM}, req)
	require.NoError(t, err)

	auth := req.Header.Get("Authorization")
	assertionJSON, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(auth, "AgentAssertion "))
	require.NoError(t, err)
	var assertion agentAssertionEnvelope
	require.NoError(t, json.Unmarshal(assertionJSON, &assertion))
	require.Equal(t, "task-789", assertion.TaskID)
	sig, err := base64.StdEncoding.DecodeString(assertion.Signature)
	require.NoError(t, err)
	require.True(t, ed25519.Verify(publicKey, []byte("runtime-123:task-789:2026-05-22T12:34:56Z"), sig))
}

func TestTransformRequestRejectsInvalidJWT(t *testing.T) {
	c := buildTestTransform(t, "not-a-jwt", "https://auth.example.test/api/accounts")
	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionReject, res.Action)
	require.Equal(t, http.StatusBadGateway, res.Response.StatusCode)
	require.Equal(t, "Bearer CODEX_ACCESS_TOKEN", req.Header.Get("Authorization"))
	require.Empty(t, req.Header.Get("ChatGPT-Account-ID"))
}

func TestTransformRequestRejectsInvalidClaims(t *testing.T) {
	_, _, token := fakeAgentIdentityTokenWithClaims(t, map[string]any{
		"iss":               expectedIssuer,
		"aud":               expectedAudience,
		"exp":               fixedNow.Add(time.Hour).Unix(),
		"agent_runtime_id":  "runtime-123",
		"agent_private_key": "not-base64",
		"account_id":        "account-abc",
	})
	c := buildTestTransform(t, token, "https://auth.example.test/api/accounts")
	c.entries[0].now = func() time.Time { return fixedNow }
	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")

	res, err := c.TransformRequest(context.Background(), &transform.TransformContext{Mode: transform.ModeMITM}, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionReject, res.Action)
	require.Equal(t, http.StatusBadGateway, res.Response.StatusCode)
}

func TestTransformRequestRejectsExpiredToken(t *testing.T) {
	publicKey, privateKey, _ := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)
	_, _, token := encodeAgentIdentityToken(t, publicKey, privateKey, map[string]any{
		"exp": fixedNow.Add(-time.Minute).Unix(),
	})
	c := buildTestTransform(t, token, "https://auth.example.test/api/accounts")
	c.entries[0].now = func() time.Time { return fixedNow }
	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")

	res, err := c.TransformRequest(context.Background(), &transform.TransformContext{Mode: transform.ModeMITM}, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionReject, res.Action)
	require.Equal(t, http.StatusBadGateway, res.Response.StatusCode)
}

func TestTransformRequestRejectsTaskRegistrationFailure(t *testing.T) {
	_, _, token := fakeAgentIdentityToken(t, "runtime-123", "account-abc", false)

	authAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "upstream failure with task-should-not-be-logged", http.StatusInternalServerError)
	}))
	defer authAPI.Close()

	c := buildTestTransform(t, token, authAPI.URL)
	req := newMatchingRequest()
	req.Header.Set("Authorization", "Bearer CODEX_ACCESS_TOKEN")
	tctx := &transform.TransformContext{Mode: transform.ModeMITM}

	res, err := c.TransformRequest(context.Background(), tctx, req)
	require.NoError(t, err)
	require.Equal(t, transform.ActionReject, res.Action)
	require.Equal(t, http.StatusBadGateway, res.Response.StatusCode)
	annotations := tctx.DrainAnnotations()
	require.Equal(t, "agent_identity_unavailable", annotations["rejected"])
	errorText, ok := annotations["error"].(string)
	require.True(t, ok)
	// Regression guard: the user-visible error must surface only resp.Status,
	// not the response body. Upstream body contents are never read or logged
	// (see registerTask) so a malicious or buggy intermediary cannot get
	// arbitrary text into iron-proxy logs by echoing it back here.
	require.NotContains(t, errorText, "task-should-not-be-logged")
}

func buildTestTransform(t *testing.T, token, authAPIBaseURL string) *CodexAuth {
	t.Helper()
	var cfg config
	raw := `
identities:
  - source:
      type: env
      var: CODEX_ACCESS_TOKEN
    authapi_base_url: ` + authAPIBaseURL + `
    rules:
      - host: chatgpt.com
        paths:
          - /backend-api/codex/*
`
	require.NoError(t, yaml.Unmarshal([]byte(raw), &cfg))
	c, err := newFromConfig(cfg, slog.Default(), staticBuilder(staticSource(token)))
	require.NoError(t, err)
	return c
}

func newMatchingRequest() *http.Request {
	return httptest.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", nil)
}

func fakeAgentIdentityToken(t *testing.T, runtimeID, accountID string, fedramp bool) (ed25519.PublicKey, ed25519.PrivateKey, string) {
	t.Helper()
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	return encodeAgentIdentityToken(t, publicKey, privateKey, map[string]any{
		"agent_runtime_id":           runtimeID,
		"account_id":                 accountID,
		"chatgpt_account_is_fedramp": fedramp,
	})
}

func encodeAgentIdentityToken(
	t *testing.T,
	publicKey ed25519.PublicKey,
	privateKey ed25519.PrivateKey,
	overrides map[string]any,
) (ed25519.PublicKey, ed25519.PrivateKey, string) {
	t.Helper()
	der, err := x509.MarshalPKCS8PrivateKey(privateKey)
	require.NoError(t, err)

	header := map[string]string{"alg": "EdDSA", "typ": "JWT"}
	payload := map[string]any{
		"iss":                        expectedIssuer,
		"aud":                        expectedAudience,
		"iat":                        1779460000,
		"exp":                        time.Now().Add(time.Hour).Unix(),
		"agent_runtime_id":           "runtime-123",
		"agent_private_key":          base64.StdEncoding.EncodeToString(der),
		"account_id":                 "account-abc",
		"chatgpt_user_id":            "user-123",
		"email":                      "test@example.com",
		"plan_type":                  "plus",
		"chatgpt_account_is_fedramp": false,
	}
	for k, v := range overrides {
		payload[k] = v
	}
	return publicKey, privateKey, encodeJWTPart(t, header) + "." + encodeJWTPart(t, payload) + ".signature"
}

func fakeAgentIdentityTokenWithClaims(t *testing.T, claims map[string]any) (ed25519.PublicKey, ed25519.PrivateKey, string) {
	t.Helper()
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	header := map[string]string{"alg": "EdDSA", "typ": "JWT"}
	return publicKey, privateKey, encodeJWTPart(t, header) + "." + encodeJWTPart(t, claims) + ".signature"
}

func encryptTaskID(t *testing.T, privateKey ed25519.PrivateKey, taskID string) string {
	t.Helper()
	var curvePrivate [32]byte
	digest := sha512.Sum512(privateKey.Seed())
	copy(curvePrivate[:], digest[:32])
	curvePrivate[0] &= 248
	curvePrivate[31] &= 127
	curvePrivate[31] |= 64

	curvePublicBytes, err := curve25519.X25519(curvePrivate[:], curve25519.Basepoint)
	require.NoError(t, err)
	var curvePublic [32]byte
	copy(curvePublic[:], curvePublicBytes)

	ciphertext, err := box.SealAnonymous(nil, []byte(taskID), &curvePublic, rand.Reader)
	require.NoError(t, err)
	return base64.StdEncoding.EncodeToString(ciphertext)
}

func encodeJWTPart(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	require.NoError(t, err)
	return base64.RawURLEncoding.EncodeToString(raw)
}
